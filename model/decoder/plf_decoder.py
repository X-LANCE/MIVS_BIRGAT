#coding=utf8
import torch
import torch.nn as nn
from model.decoder.lf_decoder import LFDecoder
from model.model_utils import Registrable, SLUHypothesisDomainWrapper
from model.decoder.lf_beam import LFBeam


@Registrable.register('plf')
class PLFDecoder(LFDecoder):

    def __init__(self, args, tranx):
        super(PLFDecoder, self).__init__(args, tranx)
        self.domain_classifier = nn.Linear(args.hidden_size, 1)
        self.domain_loss_function = nn.BCEWithLogitsLoss(reduction='sum')


    def score(self, memories, batch):
        if batch.use_domain_classifier:
            # domain classifier
            ontology_score = self.domain_classifier(memories['ontology']).squeeze(-1)
            ontology_loss = self.domain_loss_function(ontology_score.masked_select(batch.domain_mask), batch.domain_label)
        else: ontology_loss = .0

        # repeat ${domain} times for each training sample respectively
        memories['encodings'] = torch.index_select(memories['encodings'], 0, batch.repeat_indexes)
        memories['ontology'] = torch.index_select(memories['ontology'], 0, batch.repeat_indexes)
        memories['copy'] = torch.index_select(memories['copy'], 0, batch.repeat_indexes)
        memories['mask'] = torch.index_select(memories['mask'], 0, batch.repeat_indexes)
        memories['ontology_mask'] = torch.index_select(memories['ontology_mask'], 0, batch.repeat_indexes)
        memories['copy_mask'] = torch.index_select(memories['copy_mask'], 0, batch.repeat_indexes)
        memories['copy_ids'] = torch.index_select(memories['copy_ids'], 0, batch.repeat_indexes)

        loss = LFDecoder.score(self, memories, batch)
        return ontology_loss + loss


    def parse(self, memories, batch, beam_size=5, n_best=5, **kwargs):
        if batch.use_domain_classifier:
            domain_label = self.domain_classifier(memories['ontology']).squeeze(-1) >= 0
            domain_label = domain_label.masked_fill_(~ batch.domain_mask, False)
        else: domain_label = batch.domain_mask

        repeat_indexes = torch.arange(len(batch)).to(batch.device).unsqueeze(-1).expand_as(domain_label).masked_select(domain_label)
        if repeat_indexes.numel() == 0: # all examples have empty semantics
            return [SLUHypothesisDomainWrapper({}) for _ in range(len(batch))]

        memories['encodings'] = torch.index_select(memories['encodings'], 0, repeat_indexes)
        memories['ontology'] = torch.index_select(memories['ontology'], 0, repeat_indexes)
        memories['copy'] = torch.index_select(memories['copy'], 0, repeat_indexes)
        memories['mask'] = torch.index_select(memories['mask'], 0, repeat_indexes)
        memories['ontology_mask'] = torch.index_select(memories['ontology_mask'], 0, repeat_indexes)
        memories['copy_mask'] = torch.index_select(memories['copy_mask'], 0, repeat_indexes)
        memories['copy_ids'] = torch.index_select(memories['copy_ids'], 0, repeat_indexes)

        bias = self.tranx.tokenizer.vocab_size
        index = torch.arange(domain_label.numel()).reshape_as(domain_label).to(batch.device).masked_select(domain_label)
        eids, dids = index // domain_label.size(1), index % domain_label.size(1)
        beams = [LFBeam(self.tranx, batch.indexes[batch.select_domains[eid]], beam_size=beam_size, n_best=n_best, device=batch.device, bos=bias + did)
            for eid, did in zip(eids.tolist(), dids.tolist())]
        completed_hyps = LFDecoder.parse(self, memories, batch, beam_size=beam_size, n_best=n_best, beams=beams)

        # post-processing, aggregating domains for each sample
        cumulated_domain_num = [0] + torch.cumsum(domain_label.sum(dim=1), dim=0).tolist()

        def merge_different_domains(idx):
            start, end = cumulated_domain_num[idx], cumulated_domain_num[idx + 1]
            hyps = completed_hyps[start:end]
            if len(hyps) == 0: return SLUHypothesisDomainWrapper({})
            items = [(self.tranx.ontology[batch.select_domains[idx]]['domain'][did], hyp) for did, hyp in zip(dids[start: end].tolist(), hyps)]
            return SLUHypothesisDomainWrapper(items)

        wrapped_hyps = [[merge_different_domains(idx)] for idx in range(len(batch))]
        return wrapped_hyps
