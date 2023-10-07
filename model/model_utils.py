#coding=utf8
import copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class SLUHypothesis():

    __slots__ = ('action', 'score')

    def __init__(self, action: list, score: torch.FloatTensor) -> None:
        self.action = action.tolist() if torch.is_tensor(action) else action
        self.score = score.item() if torch.is_tensor(score) else score


class SLUHypothesisDomainWrapper():

    __slots__ = ('action')

    def __init__(self, items) -> None:
        """ items should be
        [(domain_name, [SLUHypothesis, SLUHypothesis, ...]), (domain_name, [SLUHypothesis, SLUHypothesis, ...])]
        """
        self.action = dict(items) # key->value: domain->ids


class SLUHypothesisClassifierWrapper():

    __slots__ = ('sl_action', 'clf_action')

    def __init__(self, sl_labels, clf_labels) -> None:
        self.sl_action = list(sl_labels)
        self.clf_action = list(clf_labels)


def shift_tokens_right(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """ Shift input ids one token to the right.
    """
    shifted_input_ids = labels.new_zeros(labels.size())
    shifted_input_ids[:, 1:] = labels[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def lens2mask(lens, max_len=None):
    bsize = lens.numel()
    max_len = lens.max() if max_len is None else max_len
    masks = torch.arange(0, max_len).type_as(lens).to(lens.device).repeat(bsize, 1).lt(lens.unsqueeze(1))
    masks.requires_grad = False
    return masks


def tile(x, count, dim=0):
    """
        Tiles x on dimension dim count times.
        E.g. [1, 2, 3], count=2 ==> [1, 1, 2, 2, 3, 3]
            [[1, 2], [3, 4]], count=3, dim=1 ==> [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]
        Different from torch.repeat
    """
    if x is None:
        return x
    elif type(x) in [list, tuple]:
        return type(x)([tile(each, count, dim) for each in x])
    else:
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.contiguous().view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x


def rnn_wrapper(encoder, inputs, lens, cell='lstm'):
    """
        @args:
            encoder(nn.Module): rnn series bidirectional encoder, batch_first=True
            inputs(torch.FloatTensor): rnn inputs, [bsize x max_seq_len x in_dim]
            lens(torch.LongTensor): seq len for each sample, allow length=0, padding with 0-vector, [bsize]
        @return:
            out(torch.FloatTensor): output of encoder, bsize x max_seq_len x hidden_dim*2
            hidden_states([tuple of ]torch.FloatTensor): final hidden states, num_layers*2 x bsize x hidden_dim
    """
    # rerank according to lens and remove empty inputs
    sorted_lens, sort_key = torch.sort(lens, descending=True)
    nonzero_num, total_num = torch.sum(sorted_lens > 0).item(), sorted_lens.size(0)
    sort_key = sort_key[:nonzero_num]
    sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key)
    # forward non empty inputs    
    packed_inputs = rnn_utils.pack_padded_sequence(sorted_inputs, sorted_lens[:nonzero_num].tolist(), batch_first=True)
    packed_out, sorted_h = encoder(packed_inputs)  # bsize x srclen x dim
    sorted_out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
    if cell.upper() == 'LSTM':
        sorted_h, sorted_c = sorted_h
    # rerank according to sort_key
    out_shape = list(sorted_out.size())
    out_shape[0] = total_num
    out = sorted_out.new_zeros(*out_shape).scatter_(0, sort_key.unsqueeze(-1).unsqueeze(-1).repeat(1, *out_shape[1:]), sorted_out)
    h_shape = list(sorted_h.size())
    h_shape[1] = total_num
    h = sorted_h.new_zeros(*h_shape).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).repeat(h_shape[0], 1, h_shape[-1]), sorted_h)
    if cell.upper() == 'LSTM':
        c = sorted_c.new_zeros(*h_shape).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).repeat(h_shape[0], 1, h_shape[-1]), sorted_c)
        return out, (h.contiguous(), c.contiguous())
    return out, h.contiguous()


class PositionalEncoding(nn.Module):
    """ Implement the PE function. """
    def __init__(self, d_model, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, timestep=0): # bs x seqlen x embed_size
        return self.dropout(self.layer_norm(x + self.pe[:, timestep: timestep + x.size(1)]))


class MultiHeadAttention(nn.Module):

    def __init__(self, q_size, kv_size, hidden_size=None, output_size=None, num_heads=8, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = int(num_heads)
        self.hidden_size = hidden_size if hidden_size is not None else q_size
        self.output_size = output_size if hidden_size is not None else q_size
        assert self.hidden_size % self.num_heads == 0, 'Head num %d must be divided by hidden size %d' % (num_heads, hidden_size)
        self.scale_factor = math.sqrt(self.hidden_size // self.num_heads)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.W_q = nn.Linear(q_size, self.hidden_size, bias=True)
        self.W_k = nn.Linear(kv_size, self.hidden_size, bias=False)
        self.W_v = nn.Linear(kv_size, self.hidden_size, bias=False)
        self.W_o = nn.Linear(self.hidden_size, self.output_size, bias=True)


    def forward(self, q_hiddens, kv_hiddens, mask=None):
        """ @params:
                q_hiddens : bsize [x tgtlen ]x hidden_size
                kv_hiddens : encoded sequence representations, bsize x srclen x hidden_size
                mask : length mask for hiddens, ByteTensor, bsize x srclen
            @return:
                context : bsize x[ tgtlen x] hidden_size
        """
        remove_flag = False
        if q_hiddens.dim() == 2:
            q_hiddens, remove_flag = q_hiddens.unsqueeze(1), True
        Q = self.W_q(self.dropout_layer(q_hiddens)).reshape(q_hiddens.size(0), q_hiddens.size(1), self.num_heads, -1)
        K = self.W_k(self.dropout_layer(kv_hiddens)).reshape(kv_hiddens.size(0), kv_hiddens.size(1), self.num_heads, -1)
        V = self.W_v(self.dropout_layer(kv_hiddens)).reshape(kv_hiddens.size(0), kv_hiddens.size(1), self.num_heads, -1)
        e = torch.einsum('bthd,bshd->btsh', Q, K) / self.scale_factor
        if mask is not None:
            e = e.masked_fill_(~ mask.unsqueeze(1).unsqueeze(-1), -1e10)
        a = torch.softmax(e, dim=2)
        concat = torch.einsum('btsh,bshd->bthd', a, V).reshape(-1, q_hiddens.size(1), self.hidden_size)
        context = self.W_o(concat)
        if remove_flag: return context.squeeze(dim=1)
        else: return context


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, hidden_size, num_heads=8, dropout=0.2):
        super(MultiHeadCrossAttention, self).__init__()
        self.hidden_size, self.num_heads = hidden_size, int(num_heads)
        self.mha = MultiHeadAttention(self.hidden_size, self.hidden_size, num_heads=self.num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, linp, rinp, lm=None, rm=None):
        lcxt = self.mha(linp, rinp, mask=rm)
        lcxt = self.layer_norm(linp + lcxt)
        rcxt = self.mha(rinp, linp, mask=lm)
        rcxt = self.layer_norm(rinp + rcxt)
        return lcxt, rcxt


class PointerNetwork(nn.Module):

    def __init__(self, q_size, k_size, hidden_size=None, num_heads=8, dropout=0.2):
        super(PointerNetwork, self).__init__()
        self.num_heads = int(num_heads)
        self.hidden_size = hidden_size if hidden_size is not None else q_size
        assert self.hidden_size % self.num_heads == 0, 'Head num %d must be divided by hidden size %d' % (num_heads, hidden_size)
        self.scale_factor = math.sqrt(self.hidden_size // self.num_heads)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.W_q = nn.Linear(q_size, self.hidden_size, bias=True)
        self.W_k = nn.Linear(k_size, self.hidden_size, bias=False)


    def forward(self, q_hiddens, k_hiddens, mask=None):
        """ @params:
                q_hiddens : bsize [x tgtlen ]x hidden_size
                k_hiddens : encoded sequence representations, bsize x srclen x hidden_size
                mask : length mask for hiddens, ByteTensor, bsize x srclen
            @return:
                attention_weight : bsize x[ tgtlen x] hidden_size
        """
        remove_flag = False
        if q_hiddens.dim() == 2:
            q_hiddens, remove_flag = q_hiddens.unsqueeze(1), True
        Q = self.W_q(self.dropout_layer(q_hiddens)).reshape(q_hiddens.size(0), q_hiddens.size(1), self.num_heads, -1)
        K = self.W_k(self.dropout_layer(k_hiddens)).reshape(k_hiddens.size(0), k_hiddens.size(1), self.num_heads, -1)
        e = torch.einsum('bthd,bshd->btsh', Q, K) / self.scale_factor
        if mask is not None:
            e = e.masked_fill_(~ mask.unsqueeze(1).unsqueeze(-1), -1e10)
        a = torch.softmax(e, dim=2)
        if remove_flag: return a.mean(dim=-1).squeeze(dim=1)
        else: return a.mean(dim=-1)


class PoolingFunction(nn.Module):
    """ Map a sequence of hidden_size dim vectors into one fixed size vector with dimension output_size """
    def __init__(self, hidden_size=256, output_size=256, bias=True, method='attentive-pooling'):
        super(PoolingFunction, self).__init__()
        assert method in ['mean-pooling', 'max-pooling', 'attentive-pooling']
        self.method = method
        if self.method == 'attentive-pooling':
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=bias),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=bias)
            )
        self.mapping_function = nn.Sequential(nn.Linear(hidden_size, output_size, bias=bias), nn.Tanh()) \
            if hidden_size != output_size else lambda x: x

    def forward(self, inputs, mask=None):
        """ @args:
                inputs(torch.FloatTensor): features, batch_size x seq_len x hidden_size
                mask(torch.BoolTensor): mask for inputs, batch_size x seq_len
            @return:
                outputs(torch.FloatTensor): aggregate seq_len dim for inputs, batch_size x output_size
        """
        if self.method == 'max-pooling':
            outputs = inputs.masked_fill(~ mask.unsqueeze(-1), -1e8)
            outputs = outputs.max(dim=1)[0]
        elif self.method == 'mean-pooling':
            mask_float = mask.float().unsqueeze(-1)
            outputs = (inputs * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        elif self.method == 'attentive-pooling':
            e = self.attn(inputs).squeeze(-1)
            e = e + (1 - mask.float()) * (-1e20)
            a = torch.softmax(e, dim=1).unsqueeze(1)
            outputs = torch.bmm(a, inputs).squeeze(1)
        else:
            raise ValueError('[Error]: Unrecognized pooling method %s !' % (self.method))
        outputs = self.mapping_function(outputs)
        return outputs


class ScoreFunction(nn.Module):

    def __init__(self, hidden_size, output_size, mlp=1, method='biaffine'):
        super(ScoreFunction, self).__init__()
        assert method in ['bilinear', 'affine', 'biaffine']
        self.mlp = int(mlp)
        self.hidden_size = hidden_size // self.mlp
        self.output_size = output_size
        if self.mlp > 1: # use mlp to perform dim reduction
            self.mlp_node = nn.Sequential(nn.Linear(hidden_size, self.hidden_size), nn.Tanh())
            self.mlp_context = nn.Sequential(nn.Linear(hidden_size, self.hidden_size), nn.Tanh())
        self.method = method
        if self.method == 'bilinear':
            self.W = nn.Bilinear(self.hidden_size, self.hidden_size, self.output_size)
        elif self.method == 'affine':
            self.affine = nn.Linear(self.hidden_size * 2, self.output_size)
        elif self.method == 'biaffine': # bilinear + affine
            self.W = nn.Bilinear(self.hidden_size, self.hidden_size, self.output_size, bias=False)
            self.affine = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, node, context):
        """
        @args:
            node(torch.FloatTensor): ... x hidden_size
            context(torch.FloatTensor): ... x hidden_size
        @return:
            scores(torch.FloatTensor): ... x output_size
        """
        if self.mlp > 1:
            node, context = self.mlp_node(node), self.mlp_context(context)
        if self.method == 'bilinear':
            scores = self.W(node, context)
        elif self.method == 'affine':
            scores = self.affine(torch.cat([node, context], dim=-1))
        elif self.method == 'biaffine':
            scores = self.W(node, context) + self.affine(torch.cat([node, context], dim=-1))
        else:
            raise ValueError('[Error]: Unrecognized score function method %s!' % (self.method))
        return scores


class FFN(nn.Module):

    def __init__(self, input_size):
        super(FFN, self).__init__()
        self.input_size = input_size
        self.feedforward = nn.Sequential(
            nn.Linear(self.input_size, self.input_size * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size * 4, self.input_size)
        )
        self.layernorm = nn.LayerNorm(self.input_size)

    def forward(self, inputs):
        return self.layernorm(inputs + self.feedforward(inputs))


class TiedLinearClassifier(nn.Module):
    def __init__(self, input_size, embed_size, bias=False):
        super(TiedLinearClassifier, self).__init__()
        self.affine = nn.Linear(input_size, embed_size, bias=bias)

    def forward(self, inputs, label_embeddings, log=True):
        outputs = self.affine(inputs)
        # assert outputs.size(-1) == label_embeddings.size(-1)
        logits = F.linear(outputs, label_embeddings)
        if log: return F.log_softmax(logits, dim=-1)
        else: return torch.softmax(logits, dim=-1)


class Registrable(object):
    """
    A class that collects all registered components,
    adapted from `common.registrable.Registrable` from AllenNLP
    """
    registered_components = dict()

    @staticmethod
    def register(name):
        def register_class(cls):
            if name in Registrable.registered_components:
                raise RuntimeError('class %s already registered' % name)

            Registrable.registered_components[name] = cls
            return cls

        return register_class

    @staticmethod
    def by_name(name):
        return Registrable.registered_components[name]


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
