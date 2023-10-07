#coding=utf8
import torch
import torch.nn as nn
from model.model_utils import Registrable, PoolingFunction


@Registrable.register('generation')
class GenerationOutputLayer(nn.Module):

    def __init__(self, args, tranx):
        super(GenerationOutputLayer, self).__init__()
        self.hidden_size = args.hidden_size


    def forward(self, questions, ontologies, batch, word_embed):
        """ Construct init decoder states, memories and masks for the decoder, including copy_memory and ontology_memory.
        @args:
            inputs~(torch.FloatTensor): bs x (max_len + max_question_len) x hs
            word_embed~(torch.FloatTensor): encoder word embedding module, vocab_size x embedding_size
        @return:
            memories~(Dict[key, torch.Tensor])
        """
        memories = {'encodings': torch.cat([questions, ontologies], dim=1), 'mask': torch.cat([batch.question_mask, batch.ontology_mask], dim=1)}
        # remove trailing slot value nodes
        memories['ontology'], memories['copy'], memories['generator'] = ontologies, questions, word_embed
        memories['ontology_mask'], memories['copy_mask'], memories['copy_ids'] = batch.ontology_mask, batch.question_mask, batch.copy_ids
        return memories


@Registrable.register('labeling')
class LabelingOutputLayer(nn.Module):

    def __init__(self, args, tranx):
        super(LabelingOutputLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.bio_label = nn.Embedding(3, self.hidden_size)
        self.affine = nn.Linear(4 * self.hidden_size, self.hidden_size) # B/I - domain - intent - slot


    def forward(self, questions, ontologies, batch, word_embed):
        """ Given the joint encodings, split into question and ontology hidden states, and construct the label embeddings of
        B/I-domain-intent-slot and special O label. Attention that domain -> intent -> slot must follow the hierarchy in the label space.
        The label embedding of B/I-domain-intent-slot is constructed by a concatenation of four vectors and a linear transformation.
        """
        # select domain-intent-slot embeddings for each label
        ontology_embeddings = [ontologies[eid][batch.label_dict[domain]].contiguous().view(-1, ontologies.size(-1) * 3) for eid, domain in enumerate(batch.select_domains)]
        ontology_embeddings = torch.cat(ontology_embeddings, dim=0)
        BIO_label = self.bio_label.weight
        BI_label = BIO_label[1:].unsqueeze(0).expand(ontology_embeddings.size(0) // 2, -1, -1).contiguous().view(-1, BIO_label.size(-1))
        label_embeddings = self.affine(torch.cat([BI_label, ontology_embeddings], dim=-1))
        # construct label embeddings
        label_mask = torch.clone(batch.label_mask)
        label_mask[:, 0] = False
        label_tensor = questions.new_zeros((len(batch), batch.label_mask.size(-1), questions.size(-1)))
        label_tensor.masked_scatter_(label_mask.unsqueeze(-1), label_embeddings)
        label_mask = batch.label_mask.new_zeros(label_mask.size())
        label_mask[:, 0] = True
        o_label = BIO_label[0].unsqueeze(0).expand(len(batch), -1)
        label_tensor.masked_scatter_(label_mask.unsqueeze(-1), o_label)
        memories = {'question': questions, 'label': label_tensor}
        return memories


@Registrable.register('labeling+classifier')
class LabelingClassifierOutputLayer(LabelingOutputLayer):

    def __init__(self, args, tranx):
        super(LabelingClassifierOutputLayer, self).__init__(args, tranx)
        self.question_pooling = PoolingFunction(args.hidden_size, args.hidden_size)


    def forward(self, questions, ontologies, batch, word_embed):
        """ Aggregate each question sequence into a single vector to perform sentence-level classification.
        """
        memories = LabelingOutputLayer.forward(self, questions, ontologies, batch, word_embed)
        memories['pooled_question'] = self.question_pooling(memories['question'], batch.question_mask)
        return memories