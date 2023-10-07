#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable, SLUHypothesis, SLUHypothesisClassifierWrapper


@Registrable.register('sl')
class SLDecoder(nn.Module):

    def __init__(self, args, tranx):
        super(SLDecoder, self).__init__()
        self.args, self.tranx = args, tranx
        self.loss_function = nn.NLLLoss(reduction='sum', ignore_index=-100)

    def forward(self, memories, batch):
        questions, labels = memories['question'], memories['label'].transpose(1, 2)
        outputs = torch.bmm(questions, labels) # bs x max_question_len x max_label_len
        outputs.masked_fill_(~(batch.question_mask.unsqueeze(-1) & batch.label_mask.unsqueeze(1)), -1e32)
        logprobs = F.log_softmax(outputs, dim=-1)
        return logprobs

    def score(self, memories, batch):
        logprobs = self(memories, batch)
        return self.loss_function(logprobs.contiguous().view(-1, batch.label_mask.size(-1)), batch.sl_tgt_actions.contiguous().view(-1))

    def parse(self, memories, batch, **kwargs):
        logprobs = self(memories, batch)
        label_scores, label_ids = torch.max(logprobs, dim=-1)
        label_scores = torch.sum(label_scores, dim=1)
        return [[SLUHypothesis(labels[:batch.question_lens[idx].item()], label_scores[idx])] for idx, labels in enumerate(label_ids)]


@Registrable.register('sl+clf')
class SLCDecoder(SLDecoder):

    def __init__(self, args, tranx):
        super(SLCDecoder, self).__init__(args, tranx)
        domain = '_cross_'.join(sorted(args.domains))
        vocab_size = len(tranx.ontology[domain]['classifier'])
        self.labels = nn.Embedding(vocab_size, args.hidden_size)
        self.classifier_loss = nn.BCEWithLogitsLoss(reduction='sum')


    def score(self, memories, batch):
        labeling_loss = SLDecoder.score(self, memories, batch)
        classifier_scores = torch.matmul(memories['pooled_question'], self.labels.weight.transpose(0, 1))
        classifier_loss = self.classifier_loss(classifier_scores.contiguous().view(-1), batch.clf_tgt_actions.contiguous().view(-1))
        return labeling_loss + classifier_loss


    def parse(self, memories, batch, **kwargs):
        hyps = SLDecoder.parse(self, memories, batch, **kwargs)
        classifier_scores = torch.matmul(memories['pooled_question'], self.labels.weight.transpose(0, 1))
        labels = (classifier_scores >= 0).int().tolist()
        return [[SLUHypothesisClassifierWrapper(hyp[0].action, l)] for hyp, l in zip(hyps, labels)]