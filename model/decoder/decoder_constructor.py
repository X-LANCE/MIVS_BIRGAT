#coding=utf8
import torch.nn as nn
from model.model_utils import Registrable
from model.decoder.lf_decoder import LFDecoder
from model.decoder.plf_decoder import PLFDecoder
from model.decoder.sl_decoder import SLDecoder, SLCDecoder


class Decoder(nn.Module):

    def __init__(self, args, tranx):
        super(Decoder, self).__init__()
        self.decoder = Registrable.by_name(args.decode_method)(args, tranx)


    def forward(self, memories, batch):
        return self.decoder.score(memories, batch)


    def parse(self, memories, batch, beam_size=5, n_best=5, **kwargs):
        return self.decoder.parse(memories, batch, beam_size=beam_size, n_best=n_best, **kwargs)