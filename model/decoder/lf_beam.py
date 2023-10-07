import torch
from model.model_utils import SLUHypothesis


class LFBeam(object):
    """ Class for managing the internals of the beam search process.
        Takes care of beams, back pointers, and scores.
        @args:
            tranx (dict): obtain indices of padding, beginning, and ending.
            beam_size (int): beam size
            n_best (int): return hyp number
            device (torch.device)
            bos (int): optional, starting symbol
    """
    def __init__(self, tranx, indexes, beam_size, n_best=1, device=None, bos=None, top_k=0):
        super(LFBeam, self).__init__()
        self.tokenizer = tranx.tokenizer
        self.shifts = (self.tokenizer.vocab_size, self.tokenizer.vocab_size + indexes[0], self.tokenizer.vocab_size + indexes[0] + indexes[1])
        self.mappings = {
            '{': self.tokenizer.convert_tokens_to_ids('{'),
            '[': self.tokenizer.convert_tokens_to_ids('['),
            '(': self.tokenizer.convert_tokens_to_ids('('),
            '}': self.tokenizer.convert_tokens_to_ids('}'),
            ']': self.tokenizer.convert_tokens_to_ids(']'),
            ')': self.tokenizer.convert_tokens_to_ids(')'),
        }
        self.beam_size, self.n_best = beam_size, n_best
        self.device = device
        # The score for each translation on the beam.
        self.scores = torch.zeros(self.beam_size, dtype=torch.float, device=self.device)

        # Has EOS topped the beam yet.
        self._eos = self.tokenizer.sep_token_id
        self.eos_top = False

        # Other special symbols
        self._bos = self.tokenizer.cls_token_id if bos is None else int(bos)
        self._pad = self.tokenizer.pad_token_id

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.zeros(self.beam_size, dtype=torch.long, device=self.device).fill_(self._pad)]
        self.next_ys[0][0] = self._bos

        # Time and k pair for finished.
        self.completed_hyps = []
        self.top_k = int(top_k) if top_k >= 2 and top_k <= self.beam_size else self.beam_size


    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]


    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]


    def advance(self, word_probs):
        """
        Given prob over words for every last beam `wordLk`

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)

        Returns: True if beam search is complete.
        """
        # CONSTRAINED DECODING:
        word_probs[:, self._pad] = -1e20
        # if len(self.prev_ks) == 0: # previous symbol is _bos -> { , [ or _eos
            # mask = torch.ones(word_probs.size(-1), dtype=torch.bool, device=self.device)
            # mask[self._eos], mask[self.mappings['{']], mask[self.mappings['[']] = False, False, False
            # word_probs.masked_fill_(mask.unsqueeze(0), -1e20)
        # else:
            # masks = []
            # for i in range(self.next_ys[-1].size(0)):
                # idx = self.next_ys[-1][i].item() # previous symbol
                # mask = torch.ones(word_probs.size(-1), dtype=torch.bool, device=self.device)
                # if idx == self.mappings['{']: # next must be domain
                    # mask[self.shifts[0]: self.shifts[1]] = False
                # elif idx == self.mappings['}']: # next must be { or eos
                    # mask[self.mappings['{']], mask[self._eos] = False, False
                # elif idx == self.mappings['[']: # next must be intent
                    # mask[self.shifts[1]: self.shifts[2]] = False
                # elif idx == self.mappings[']']: # next must be [ , } or _eos
                    # mask[self.mappings['[']], mask[self.mappings['}']], mask[self._eos] = False, False, False
                # elif idx == self.mappings['(']: # next must be slot
                    # mask[self.shifts[2]:] = False
                # elif idx == self.mappings[')']: # next must be ( or ]
                    # mask[self.mappings['(']], mask[self.mappings[']']] = False, False
                # elif self.shifts[0] <= idx < self.shifts[1]: # domain -> [ or domain -> }
                    # mask[self.mappings['[']], mask[self.mappings['}']], mask[self._eos] = False, False, False
                # elif self.shifts[1] <= idx < self.shifts[2]: # intent -> ( or intent -> ]
                    # mask[self.mappings['(']], mask[self.mappings[']']] = False, False
                # else: # slot -> tokens, tokens -> tokens
                    # mask[:self.shifts[0]] = False
                # masks.append(mask)
            # word_probs.masked_fill_(torch.stack(masks, dim=0), -1e20)

        # pick top_k candidates
        cur_top_k = self.beam_size if len(self.prev_ks) == 0 else self.top_k
        top_k, sort_key = word_probs.topk(cur_top_k, -1, True, True)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = top_k + self.scores.unsqueeze(1)
        else:
            beam_scores = top_k[0]
        flat_beam_scores = beam_scores.contiguous().view(-1)
        _, best_scores_id = flat_beam_scores.topk(self.beam_size, 0, True, True)

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // cur_top_k
        self.prev_ks.append(prev_k)
        next_y = torch.take(sort_key.contiguous().view(-1), best_scores_id)
        self.next_ys.append(next_y)
        self.scores = torch.take(beam_scores.contiguous().view(-1), best_scores_id)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i].item() == self._eos:
                self.completed_hyps.append((self.scores[i], len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.eos_top = True

        return self.done


    @property
    def done(self):
        return self.eos_top and len(self.completed_hyps) >= self.n_best


    def sort_finished(self):
        if len(self.completed_hyps) > 0:
            self.completed_hyps.sort(key=lambda a: - a[0]) # / a[1])
            completed_hyps = [SLUHypothesis(action=self.get_hyp(t, k), score=s) for s, t, k in self.completed_hyps]
        else:
            completed_hyps = [SLUHypothesis(action=[self._eos], score=-float('inf'))]
        return completed_hyps


    def get_hyp(self, timestep, k):
        """ Walk back to construct the full hypothesis. 
            hyp contains [SEP] but does not contain [CLS]
            @return:
                hyp: list of id
        """
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k].item())
            k = self.prev_ks[j][k]
        return hyp[::-1]
