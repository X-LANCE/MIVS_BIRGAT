#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable, PointerNetwork, TiedLinearClassifier, PositionalEncoding, tile
from model.decoder.lf_beam import LFBeam


@Registrable.register('lf')
class LFDecoder(nn.Module):

    def __init__(self, args, tranx):
        super(LFDecoder, self).__init__()
        self.args, self.tranx = args, tranx
        self.pe = PositionalEncoding(args.hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(args.hidden_size, args.num_heads, dim_feedforward=args.hidden_size * 4, dropout=args.dropout)
        self.decoder_network = nn.TransformerDecoder(decoder_layer, args.decoder_num_layers)
        self.word_embed_affine = nn.Linear(args.embed_size, args.hidden_size) if args.embed_size != args.hidden_size else lambda x: x
        # copy raw question words and ontology items
        self.pointer_network = PointerNetwork(args.hidden_size, args.hidden_size, num_heads=args.num_heads, dropout=args.dropout)
        self.generator = TiedLinearClassifier(args.hidden_size, args.hidden_size)
        self.switcher = nn.Linear(args.hidden_size * 2, 2)
        self.loss_function = nn.NLLLoss(reduction='sum', ignore_index=self.tranx.tokenizer.pad_token_id)


    def score(self, memories, batch):
        args = self.args
        encodings, ontology_memory, generator_memory, copy_memory = memories['encodings'], memories['ontology'], memories['generator'], memories['copy']
        mask, ontology_mask, copy_mask, copy_ids = memories['mask'], memories['ontology_mask'], memories['copy_mask'], memories['copy_ids']
        generator_memory = self.word_embed_affine(generator_memory)
        # construct input matrices, bs x tgt_len x hs
        inp_actions = batch.lf_tgt_actions[:, :-1] # input is shifted
        vocab_size = self.tranx.tokenizer.vocab_size
        # tgt_ontology_mask = inp_actions >= vocab_size # bs x max_tgt_len
        # word_inputs = F.embedding(inp_actions.masked_fill(tgt_ontology_mask, 0), generator_memory) # bs x max_tgt_len x hs
        # shift_ontology_actions = inp_actions - vocab_size + ontology_mask.size(1) * torch.arange(inp_actions.size(0), device=batch.device).unsqueeze(1)
        # tgt_ontology_actions = shift_ontology_actions.masked_select(tgt_ontology_mask)
        # ontology_inputs = ontology_memory.contiguous().view(-1, args.hidden_size)[tgt_ontology_actions]
        # word_inputs = word_inputs.masked_scatter_(tgt_ontology_mask.unsqueeze(-1), ontology_inputs)
        word_inputs = F.embedding(inp_actions, generator_memory)
        decoder_inputs = self.pe(word_inputs) # add cosine position embeddings

        future_mask = torch.tril(mask.new_ones((decoder_inputs.size(1), decoder_inputs.size(1))))
        outputs = self.decoder_network(decoder_inputs.transpose(0, 1), encodings.transpose(0, 1), tgt_mask=~future_mask,
            tgt_key_padding_mask=~batch.tgt_mask, memory_key_padding_mask=~mask).transpose(0, 1) # bs x max_tgt_len x hs

        gate = torch.softmax(self.switcher(torch.cat([outputs, decoder_inputs], dim=-1)), dim=-1)
        gen_token_prob = self.generator(outputs, generator_memory, log=False) * gate[:, :, 0:1] # bs x max_tgt_len x vocab_size
        copy_token_prob = self.pointer_network(outputs, copy_memory, mask=copy_mask) * gate[:, :, 1:2] # bs x max_tgt_len x max_question_len
        copy_ids = copy_ids.unsqueeze(1).expand(-1, copy_token_prob.size(1), -1)
        copy_gen_token_prob = gen_token_prob.scatter_add_(-1, copy_ids, copy_token_prob)
        # select_ontology_prob = self.pointer_network(outputs, ontology_memory, mask=ontology_mask) * gate[:, :, 2:] # bs x max_tgt_len x max_ontology_len
        # output_prob = torch.cat([copy_gen_token_prob, select_ontology_prob], dim=-1).contiguous().view(-1, vocab_size + ontology_mask.size(1)) # bs x max_tgt_len x (vocab_size + max_ontology_len)
        output_prob = copy_gen_token_prob.contiguous().view(-1, vocab_size)
        out_actions = batch.lf_tgt_actions[:, 1:].contiguous().view(-1)
        loss = self.loss_function(torch.log(output_prob + 1e-32), out_actions)
        return loss


    def parse(self, memories, batch, beam_size=5, n_best=5, beams=None):
        args = self.args
        beams = [LFBeam(self.tranx, batch.indexes[d], beam_size, n_best, batch.device) for d in batch.select_domains] if beams is None else beams
        encodings, ontology_memory, generator_memory, copy_memory = memories['encodings'], memories['ontology'], memories['generator'], memories['copy']
        mask, ontology_mask, copy_mask, copy_ids = memories['mask'], memories['ontology_mask'], memories['copy_mask'], memories['copy_ids']
        generator_memory = self.word_embed_affine(generator_memory)
        vocab_size = self.tranx.tokenizer.vocab_size

        # tile beam_size times
        num_samples, batch_idx = encodings.size(0), list(range(encodings.size(0)))
        encodings, ontology_memory, copy_memory = tile([encodings, ontology_memory, copy_memory], beam_size)
        mask, ontology_mask, copy_mask, copy_ids = tile([mask, ontology_mask, copy_mask, copy_ids], beam_size)
        prev_inputs = encodings.new_zeros((num_samples * beam_size, 0, args.hidden_size))

        for t in range(batch.max_action_num):
            # (a) construct inputs from remaining samples
            ys = torch.cat([b.get_current_state() for b in beams if not b.done], dim=0) # num_samples * beam_size
            # ys_ontology_mask = ys >= vocab_size # num_samples * beam_size
            # if torch.any(ys_ontology_mask).item():
                # inputs = F.embedding(ys.masked_fill(ys_ontology_mask, 0), generator_memory) # num_samples * beam_size x hs
                # shift_ontology_actions = ys - vocab_size + ontology_mask.size(1) * torch.arange(num_samples * beam_size, device=batch.device)
                # tgt_ontology_actions = shift_ontology_actions.masked_select(ys_ontology_mask)
                # ontology_inputs = ontology_memory.contiguous().view(-1, args.hidden_size)[tgt_ontology_actions]
                # inputs = inputs.masked_scatter_(ys_ontology_mask.unsqueeze(-1), ontology_inputs)
            # else: inputs = F.embedding(ys, generator_memory) # num_samples * beam_size x hs
            inputs = F.embedding(ys, generator_memory) # num_samples * beam_size x hs

            # (b) calculate logprob distribution over each hyp
            decoder_inputs = self.pe(inputs.unsqueeze(1), timestep=t)
            prev_inputs = torch.cat([prev_inputs, decoder_inputs], dim=1) # num_hyps x cur_tgt_len x embed_size
            future_mask = torch.tril(mask.new_ones((t + 1, t + 1)))
            outputs = self.decoder_network(prev_inputs.transpose(0, 1), encodings.transpose(0, 1), tgt_mask=~ future_mask,
                memory_key_padding_mask=~ mask).transpose(0, 1)[:, -1] # num_hyps x hs

            gate = torch.softmax(self.switcher(torch.cat([outputs, decoder_inputs.squeeze(1)], dim=-1)), dim=-1)
            gen_token_prob = self.generator(outputs, generator_memory, log=False) * gate[:, 0:1] # num_hyps x vocab_size
            copy_token_prob = self.pointer_network(outputs, copy_memory, mask=copy_mask) * gate[:, 1:2] # num_hyps x max_question_len
            copy_gen_token_prob = gen_token_prob.scatter_add_(-1, copy_ids, copy_token_prob)
            # select_ontology_prob = self.pointer_network(outputs, ontology_memory, mask=ontology_mask) * gate[:, 2:] # num_hyps x max_ontology_len
            # output_logprob = torch.log(torch.cat([copy_gen_token_prob, select_ontology_prob], dim=-1) + 1e-32) # num_hyps x (vocab_size + max_ontology_len)
            # output_logprob = output_logprob.contiguous().view(num_samples, beam_size, vocab_size + ontology_mask.size(1))
            output_logprob = torch.log(copy_gen_token_prob + 1e-32).contiguous().view(num_samples, beam_size, vocab_size)

            # (c) advance each beam
            active, select_indexes = [], []
            # Loop over the remaining_batch number of beam
            for b in range(num_samples):
                idx = batch_idx[b] # idx represent the original order in minibatch_size
                beams[idx].advance(output_logprob[b])
                if not beams[idx].done:
                    active.append((idx, b))
                select_indexes.append(beams[idx].get_current_origin() + b * beam_size)

            if not active:
                break

            # (d) update hidden_states history
            select_indexes = torch.cat(select_indexes, dim=0)
            prev_inputs = prev_inputs[select_indexes]

            # (e) reserve un-finished batches
            active_idx = torch.tensor([item[1] for item in active], dtype=torch.long, device=batch.device) # original order in remaining batch
            batch_idx = { idx: item[0] for idx, item in enumerate(active) } # order for next remaining batch

            def update_active(inp, dim=0):
                if dim != 0: inp = inp.transpose(0, dim)
                inp_reshape = inp.contiguous().view(num_samples, beam_size, -1)[active_idx]
                new_size = list(inp.size())
                new_size[0] = -1
                inp_reshape = inp_reshape.contiguous().view(*new_size)
                if dim != 0: inp_reshape = inp_reshape.transpose(0, dim)
                return inp_reshape

            if len(active) < num_samples:
                # encodings, ontology_memory, copy_memory = update_active(encodings), update_active(ontology_memory), update_active(copy_memory)
                # mask, ontology_mask, copy_mask, copy_ids = update_active(mask), update_active(ontology_mask), update_active(copy_mask), update_active(copy_ids)
                encodings, copy_memory = update_active(encodings), update_active(copy_memory)
                mask, copy_mask, copy_ids = update_active(mask), update_active(copy_mask), update_active(copy_ids)
                prev_inputs = update_active(prev_inputs)

            num_samples = len(active)

        completed_hyps = [b.sort_finished() for b in beams]
        return completed_hyps
