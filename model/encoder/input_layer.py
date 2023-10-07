#coding=utf8
import os
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from model.model_utils import Registrable, rnn_wrapper, PoolingFunction
from utils.configs import DATASETS

@Registrable.register('plm')
class PLMInputLayer(nn.Module):

    def __init__(self, args, tranx):
        super(PLMInputLayer, self).__init__()
        plm = os.path.join(DATASETS['plm_dir'], args.plm)
        config = AutoConfig.from_pretrained(plm)
        args.embed_size = config.embedding_size if hasattr(config, 'embedding_size') else config.hidden_size
        self.plm = AutoModel.from_config(config) if getattr(args, 'lazy_load', False) else AutoModel.from_pretrained(plm)
        self.question_rnn = nn.LSTM(config.hidden_size, args.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.ontology_encoding, self.use_value = args.ontology_encoding, args.use_value
        if self.ontology_encoding: # use a separate plm for ontology items
            self.ontology_rnn = nn.LSTM(config.hidden_size, args.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        else: # directly initalize from embedding matrix, need to specify the required domains in program arguments
            vocab = tranx.ontology['_cross_'.join(sorted(args.domains))]
            self.vocab_size = len(vocab['domain']) + len(vocab['intent']) + len(vocab['slot'])
            self.ontology_embed = nn.Embedding(self.vocab_size, args.hidden_size)
        if self.use_value: # aggregate multiple slot values into one node
            self.value_pooling = PoolingFunction(args.hidden_size, args.hidden_size, method='mean-pooling')


    def forward(self, batch):
        """
        @retrun:
            q_out: torch.FloatTensor, encoded question repr, bs x max_question_len x hs
            o_out: Dict[str, torch.FloatTensor], encoded ontology repr, domain_name~(str, e.g., 打电话_cross_音乐) -> tensor~(ontology_num x hs)
        """
        q_out = self.plm(**batch.question_inputs)[0]
        q_out, _ = rnn_wrapper(self.question_rnn, q_out, batch.question_lens)

        v_out = None
        if self.ontology_encoding:
            o_out = self.plm(**batch.ontology_inputs)[0] # for each domain, flattened domain tokens + intent tokens + slot tokens
            mask = batch.ontology_token_mask.unsqueeze(-1)
            source = o_out.masked_select(batch.ontology_plm_mask.unsqueeze(-1)) # remove [PAD], [CLS] and [SEP]
            o_out = o_out.new_zeros((mask.size(0), mask.size(1), o_out.size(-1))).masked_scatter_(mask, source)
            _, hiddens = rnn_wrapper(self.ontology_rnn, o_out, batch.ontology_token_lens)
            dim = hiddens[0].size(-1) * 2
            hiddens = hiddens[0].transpose(0, 1).contiguous().view(-1, dim) # sum_ontology_items x hs
            o_out = {domain: hiddens[batch.select_ontology_indexes[domain]] for domain in batch.select_ontology_indexes}
            if self.use_value:
                values = hiddens[batch.value_indexes]
                value_num, value_len = batch.value_mask.size(0), batch.value_mask.size(1)
                values_out = values.new_zeros((value_num, value_len, values.size(-1)))
                values_out.masked_scatter_(batch.value_mask.unsqueeze(-1), values)
                values_out = self.value_pooling(values_out, batch.value_mask)
                v_out = {domain: values_out[batch.select_value_indexes[domain]] for domain in batch.select_value_indexes}
        else: # domain -> num_ontology_items x hs
            o_out = {domain: self.ontology_embed(batch.select_ontology_indexes[domain]) for domain in batch.select_ontology_indexes}
        return q_out, o_out, v_out


@Registrable.register('swv')
class SWVInputLayer(nn.Module):

    def __init__(self, args, tranx) -> None:
        super(SWVInputLayer, self).__init__()
        plm = os.path.join(DATASETS['plm_dir'], args.plm)
        # although not use PLM, re-use the word embeddings layer to reduce coding work
        config = AutoConfig.from_pretrained(plm)
        args.embed_size = config.embedding_size if hasattr(config, 'embedding_size') else config.hidden_size
        self.swv = AutoModel.from_config(config).embeddings.word_embeddings if getattr(args, 'lazy_load', False) else AutoModel.from_pretrained(plm).embeddings.word_embeddings
        self.question_rnn = nn.LSTM(args.embed_size, args.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.ontology_encoding, self.use_value = args.ontology_encoding, args.use_value
        if self.ontology_encoding:
            self.ontology_rnn = nn.LSTM(args.embed_size, args.hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        else: # directly initalize from embedding matrix, need to specify the required domains in program arguments
            domain = tranx.ontology['_cross_'.join(sorted(args.domains))]
            self.vocab_size = len(domain['domain']) + len(domain['intent']) + len(domain['slot'])
            self.ontology_embed = nn.Embedding(self.vocab_size, args.hidden_size)
        if self.use_value:
            self.value_pooling = PoolingFunction(args.hidden_size, args.hidden_size, method='mean-pooling')


    def forward(self, batch):
        """
        @retrun:
            q_out: torch.FloatTensor, encoded question repr, bs x max_question_len x hs
            o_out: Dict[str, torch.FloatTensor], encoded ontology repr, domain_name~(str) -> tensor~(ontology_num x hs)
        """
        q_out = self.swv(batch.question_inputs)
        q_out, _ = rnn_wrapper(self.question_rnn, q_out, batch.question_lens) # bs x max_qlen x hs

        v_out = None
        if self.ontology_encoding:
            o_out = self.swv(batch.ontology_inputs) # sum_ontology_items x max_ontology_token_len x embed_size
            _, hiddens = rnn_wrapper(self.ontology_rnn, o_out, batch.ontology_token_lens)
            dim = hiddens[0].size(-1) * 2
            hiddens = hiddens[0].transpose(0, 1).contiguous().view(-1, dim) # sum_ontology_items x hs
            o_out = {domain: hiddens[batch.select_ontology_indexes[domain]] for domain in batch.select_ontology_indexes}
            if self.use_value:
                values = hiddens[batch.value_indexes]
                value_num, value_len = batch.value_mask.size(0), batch.value_mask.size(1)
                values_out = values.new_zeros((value_num, value_len, values.size(-1)))
                values_out.masked_scatter_(batch.value_mask.unsqueeze(-1), values)
                values_out = self.value_pooling(values_out, batch.value_mask)
                v_out = {domain: values_out[batch.select_value_indexes[domain]] for domain in batch.select_value_indexes}
        else:
            o_out = {domain: self.ontology_embed(batch.select_ontology_indexes[domain]) for domain in batch.select_ontology_indexes}
        return q_out, o_out, v_out
