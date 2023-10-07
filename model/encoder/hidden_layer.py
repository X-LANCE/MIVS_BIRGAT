#coding=utf8
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable, clones, MultiHeadCrossAttention
from utils.configs import RELATIONS


def merge_slot_value_nodes(o_out, v_out, batch):
    outputs = []
    max_len = batch.ontology_full_mask.size(1)
    for d in batch.select_domains:
        o = o_out[d]
        if v_out is None:
            padding = o.new_zeros((max_len - o.size(0), o.size(-1)))
            outputs.append(torch.cat([o, padding], dim=0))
        else:
            v = v_out[d]
            padding = o.new_zeros((max_len - o.size(0) - v.size(0), o.size(-1)))
            outputs.append(torch.cat([o, v, padding], dim=0))
    outputs = torch.stack(outputs, dim=0) # input tensor format transformation
    return outputs


@Registrable.register('rgat')
class RGATHiddenLayer(nn.Module):

    def __init__(self, args, tranx):
        super(RGATHiddenLayer, self).__init__()
        hs, hd, rn = args.hidden_size, args.num_heads, len(RELATIONS)
        relation_pad_idx = RELATIONS.index('padding-relation')
        self.num_layers = nl = args.encoder_num_layers
        self.relation_embed_k = nn.Embedding(rn, hs // hd, padding_idx=relation_pad_idx)
        self.relation_embed_v = nn.Embedding(rn, hs // hd, padding_idx=relation_pad_idx)
        if args.cross_attention == 'layerwise' or (args.cross_attention == 'final' and nl == 1):
            gnn_module = Registrable.by_name('rgat_layer')(hs, rn, hd, dropout=args.dropout, cross_attention=True)
            self.gnn_layers = clones(gnn_module, self.num_layers)
        elif args.cross_attention == 'final':
            gnn_module = Registrable.by_name('rgat_layer')(hs, rn, hd, dropout=args.dropout, cross_attention=False)
            self.gnn_layers = clones(gnn_module, nl - 1)
            self.gnn_layers.append(Registrable.by_name('rgat_layer')(hs, rn, hd, dropout=args.dropout, cross_attention=True))
        elif args.cross_attention == 'none':
            gnn_module = Registrable.by_name('rgat_layer')(hs, rn, hd, dropout=args.dropout, cross_attention=False)
            self.gnn_layers = clones(gnn_module, nl)
        else: raise ValueError('Not recognized cross attention function for the graph encoder.')


    def forward(self, q_out, o_out, v_out, batch):
        """ Jointly encode question nodes and ontology nodes via Relational Graph Attention Network
        @args:
            q_out: torch.FloatTensor, encoded question repr, bs x max_question_len x hs
            o_out: Dict[str, torch.FloatTensor], encoded ontology repr, domain_name~(str) -> tensor~(ontology_num x hs)
            v_out: Dict[str, torch.FloatTensor], encoded slot value repr, domain_name~(str) -> tensor~(slot_num x hs)
        @return:
            outputs: torch.FloatTensor, bs x max_len x hs, max_len is sum of the maximum of question_nodes and ontology_items
        """
        o_out = merge_slot_value_nodes(o_out, v_out, batch)
        for i in range(self.num_layers):
            q_out, o_out = self.gnn_layers[i](q_out, o_out, batch, self.relation_embed_k, self.relation_embed_v)
        o_out = o_out[:, :batch.ontology_mask.size(1)].masked_fill(~ batch.ontology_mask.unsqueeze(-1), 0.) # remove value nodes
        return q_out, o_out


@Registrable.register('rgat_layer')
class RGATLayer(nn.Module):
    """ Encode question nodes and ontology nodes via relational graph attention network, parameters are shared for these two types.
    """
    def __init__(self, hidden_size=512, relation_num=len(RELATIONS), num_heads=8, feedforward=None, dropout=0.2, cross_attention=False):
        super(RGATLayer, self).__init__()
        assert hidden_size % num_heads == 0, 'Hidden size is not divisible by num of heads'
        self.hidden_size, self.relation_num, self.num_heads = hidden_size, relation_num, num_heads
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.scale_factor = math.sqrt(self.hidden_size // self.num_heads)
        self.concat_affine = nn.Linear(self.hidden_size, self.hidden_size)
        self.cross_attention = cross_attention
        if self.cross_attention:
            self.mhca = MultiHeadCrossAttention(self.hidden_size, self.num_heads, dropout=dropout)
        feedforward = self.hidden_size * 4 if feedforward is None else feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward, self.hidden_size)
        )
        self.layernorm_1 = nn.LayerNorm(self.hidden_size)
        self.layernorm_2 = nn.LayerNorm(self.hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, q, o, batch, relation_embed_k=None, relation_embed_v=None):
        qrel, qrel_mask = batch.question_relations, batch.question_relations_mask
        orel, orel_mask = batch.ontology_relations, batch.ontology_relations_mask

        def calculate_outputs(inputs, rel, mask):
            bs, l = inputs.size(0), inputs.size(1)
            q, k, v = torch.chunk(self.qkv(self.dropout_layer(inputs)), 3, dim=-1)
            q = q.view(bs, l, self.num_heads, -1).transpose(1, 2).unsqueeze(3) # q: bsize x num_heads x seqlen x 1 x dim
            # k and v: bsize x num_heads x seqlen x seqlen x dim
            k = k.view(bs, l, self.num_heads, -1).transpose(1, 2).unsqueeze(2).expand(bs, self.num_heads, l, l, -1)
            v = v.view(bs, l, self.num_heads, -1).transpose(1, 2).unsqueeze(2).expand(bs, self.num_heads, l, l, -1)
            rel_k = relation_embed_k(rel).unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
            rel_v = relation_embed_v(rel).unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)
            k, v = k + rel_k, v + rel_v

            # e: bsize x heads x seqlen x seqlen
            e = (torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor).squeeze(-2)
            if mask is not None:
                e = e.masked_fill_(mask.unsqueeze(1), -1e20) # mask no-relation
            a = torch.softmax(e, dim=-1)
            outputs = torch.matmul(a.unsqueeze(-2), v).squeeze(-2)
            outputs = outputs.transpose(1, 2).contiguous().view(bs, l, -1)
            outputs = self.concat_affine(outputs)
            outputs = self.layernorm_1(inputs + outputs)
            return outputs

        q_out, o_out = calculate_outputs(q, qrel, qrel_mask), calculate_outputs(o, orel, orel_mask)
        if self.cross_attention:
            # use batch.ontology_full_mask instead of batch.ontology_mask due to possible slot value nodes
            q_out, o_out = self.mhca(q_out, o_out, batch.question_mask, batch.ontology_full_mask)
        q_out = self.layernorm_2(q_out + self.feedforward(q_out))
        o_out = self.layernorm_2(o_out + self.feedforward(o_out))
        return q_out, o_out


@Registrable.register('gat')
class GATHiddenLayer(nn.Module):

    def __init__(self, args, tranx):
        super(GATHiddenLayer, self).__init__()
        self.num_layers = nl = args.encoder_num_layers
        params = (args.hidden_size, args.num_heads, args.hidden_size * 4, args.dropout)
        if args.cross_attention == 'layerwise' or (args.cross_attention == 'final' and nl == 1):
            gnn_module = Registrable.by_name('gat_layer')(*params, cross_attention=True)
            self.gnn_layers = clones(gnn_module, self.num_layers)
        elif args.cross_attention == 'final':
            gnn_module = Registrable.by_name('gat_layer')(*params, cross_attention=False)
            self.gnn_layers = clones(gnn_module, nl - 1)
            self.gnn_layers.append(Registrable.by_name('gat_layer')(*params, cross_attention=True))
        elif args.cross_attention == 'none':
            gnn_module = Registrable.by_name('gat_layer')(*params, cross_attention=False)
            self.gnn_layers = clones(gnn_module, nl)
        else: raise ValueError('Not recognized cross attention function for the graph encoder.')


    def forward(self, q_out, o_out, v_out, batch):
        o_out = merge_slot_value_nodes(o_out, v_out, batch)
        for i in range(self.num_layers):
            q_out, o_out = self.gnn_layers[i](q_out, o_out, batch)
        o_out = o_out[:, :batch.ontology_mask.size(1)].masked_fill(~ batch.ontology_mask.unsqueeze(-1), 0.) # remove value nodes
        return q_out, o_out


@Registrable.register('gat_layer')
class GATLayer(nn.Module):

    def __init__(self, hidden_size=512, num_heads=8, feedforward=2048, dropout=0.2, cross_attention=False):
        super(GATLayer, self).__init__()
        assert hidden_size % num_heads == 0, 'Hidden size is not divisible by num of heads'
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.scale_factor = math.sqrt(self.hidden_size // self.num_heads)
        self.concat_affine = nn.Linear(self.hidden_size, self.hidden_size)
        self.cross_attention = cross_attention
        if self.cross_attention:
            self.mhca = MultiHeadCrossAttention(self.hidden_size, self.num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward, self.hidden_size)
        )
        self.layernorm_1 = nn.LayerNorm(self.hidden_size)
        self.layernorm_2 = nn.LayerNorm(self.hidden_size)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, q, o, batch):
        """ The official implementaion of Tranformer module is complicated regarding the usage of mask matrix.
        Re-implement it with self-defined modules.
        """
        qmask, omask = batch.question_relations_mask, batch.ontology_relations_mask

        def calculate_outputs(inputs, mask):
            bsize, seqlen = inputs.size(0), inputs.size(1)
            q, k, v = torch.chunk(self.qkv(self.dropout_layer(inputs)), 3, dim=-1)
            q = q.view(bsize, seqlen, self.num_heads, -1).transpose(1, 2) # bsize x num_heads x seqlen x dim
            # bsize x num_heads x seqlen x dim
            k = k.view(bsize, seqlen, self.num_heads, -1).transpose(1, 2)
            v = v.view(bsize, seqlen, self.num_heads, -1).transpose(1, 2)
            # e: bsize x num_heads x seqlen x seqlen
            e = (torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor)
            if mask is not None:
                e = e.masked_fill_(mask.unsqueeze(1), -1e20) # mask padding-relation
            a = torch.softmax(e, dim=-1)
            outputs = torch.matmul(a, v)
            outputs = outputs.transpose(1, 2).contiguous().view(bsize, seqlen, -1)
            outputs = self.concat_affine(outputs)
            outputs = self.layernorm_1(inputs + outputs)
            return outputs

        q_out, o_out = calculate_outputs(q, qmask), calculate_outputs(o, omask)
        if self.cross_attention:
            # use batch.ontology_full_mask instead of batch.ontology_mask due to possible slot value nodes
            q_out, o_out = self.mhca(q_out, o_out, batch.question_mask, batch.ontology_full_mask)
        q_out = self.layernorm_2(q_out + self.feedforward(q_out))
        o_out = self.layernorm_2(o_out + self.feedforward(o_out))
        return q_out, o_out


@Registrable.register('none')
class NoneGraphHiddenLayer(nn.Module):

    def __init__(self, args, tranx):
        super(NoneGraphHiddenLayer, self).__init__()
        self.hidden_size, self.num_heads, self.num_layers, self.dropout = args.hidden_size, args.num_heads, args.encoder_num_layers, args.dropout
        assert self.hidden_size % self.num_heads == 0, 'Hidden size is not divisible by num of heads'
        self.init_method = args.init_method
        if self.init_method == 'swv':
            encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, self.num_heads, self.hidden_size * 4, dropout=self.dropout)
            self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

    def forward(self, q_out, o_out, v_out, batch):
        o_out = merge_slot_value_nodes(o_out, v_out, batch)
        if self.init_method == 'swv': # if init_method for question is static word vectors, use Transformer Encoder to encode contents
            q_out = self.transformer(q_out.transpose(0, 1), src_key_padding_mask=~ batch.question_mask).transpose(0, 1)
        o_out = o_out[:, :batch.ontology_mask.size(1)].masked_fill(~ batch.ontology_mask.unsqueeze(-1), 0.) # remove value nodes
        return q_out, o_out
