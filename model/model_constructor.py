#coding=utf8
import os
import torch.nn as nn
from model.model_utils import Registrable, shift_tokens_right, SLUHypothesis
from model.encoder.encoder_constructor import Encoder
from model.decoder.decoder_constructor import Decoder
from utils.configs import DATASETS
from transformers import BartForConditionalGeneration


@Registrable.register('encoder-decoder')
class EncoderDecoder(nn.Module):

    def __init__(self, args, tranx):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(args, tranx)
        self.decoder = Decoder(args, tranx)


    def forward(self, batch, *args):
        """ This function is used during training, which returns the training loss
        """
        return self.decoder(self.encoder(batch), batch)


    def parse(self, batch, beam_size=5, n_best=5, **kwargs):
        """ This function is used for decoding, which returns a batch of hypothesis
        """
        return self.decoder.parse(self.encoder(batch), batch, beam_size=beam_size, n_best=n_best, **kwargs)


@Registrable.register('bart-generation')
class BARTGenerationModel(nn.Module):

    def __init__(self, args, tranx):
        super(BARTGenerationModel, self).__init__()
        assert 'bart' in args.plm, 'Currently, we only support BART model as the generative PLM .'
        plm = os.path.join(DATASETS['plm_dir'], args.plm)
        self.bart_model = BartForConditionalGeneration.from_pretrained(plm)
        self.ontology_copy = (args.decode_method == 'gplm+copy')
        if self.ontology_copy:
            # add ontologies into the vocabulary
            vocab = tranx.ontology['_cross_'.join(sorted(args.domains))]
            self.vocab_size = self.bart_model.config.vocab_size + len(vocab['domain']) + len(vocab['intent']) + len(vocab['slot'])
            self.bart_model.resize_token_embeddings(self.vocab_size)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')


    def forward(self, batch, *args):
        input_ids, attn_mask = batch.question_inputs['input_ids'], batch.question_inputs['attention_mask']
        decoder_input_ids = shift_tokens_right(batch.lf_tgt_actions, self.bart_model.config.pad_token_id, self.bart_model.config.decoder_start_token_id)
        output_logits = self.bart_model(input_ids, attention_mask=attn_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=batch.tgt_mask)[0]
        return self.loss_function(output_logits.contiguous().view(-1, output_logits.size(-1)), batch.lf_tgt_actions.contiguous().view(-1))


    def parse(self, batch, beam_size=5, n_best=5, **kwargs):
        outputs = self.bart_model.generate(batch.question_inputs['input_ids'], num_beams=beam_size, num_return_sequences=n_best,
            max_length=batch.max_action_num, output_scores=True, return_dict_in_generate=True)
        predictions, scores = outputs.sequences.view(len(batch), n_best, -1), outputs.sequences_scores.view(len(batch), n_best)
        completed_hyps = [[SLUHypothesis(action=pred, score=score) for pred, score in zip(preds, beam_scores)] for preds, beam_scores in zip(predictions, scores)]
        return completed_hyps
