#coding=utf8
import torch.nn as nn
from model.model_utils import Registrable
from model.encoder.input_layer import PLMInputLayer, SWVInputLayer
from model.encoder.hidden_layer import RGATHiddenLayer, GATHiddenLayer, NoneGraphHiddenLayer
from model.encoder.output_layer import LabelingOutputLayer, LabelingClassifierOutputLayer, GenerationOutputLayer


class Encoder(nn.Module):

    def __init__(self, args, tranx):
        super(Encoder, self).__init__()
        self.input_layer = Registrable.by_name(args.init_method)(args, tranx)
        self.hidden_layer = Registrable.by_name(args.encode_method)(args, tranx)
        self.output_layer = Registrable.by_name('labeling')(args, tranx) if args.decode_method == 'sl' else \
            Registrable.by_name('labeling+classifier')(args, tranx) if args.decode_method == 'sl+clf' else \
            Registrable.by_name('generation')(args, tranx)

    def forward(self, batch):
        q_out, o_out, v_out = self.input_layer(batch)
        q_out, o_out = self.hidden_layer(q_out, o_out, v_out, batch)
        word_embed = self.input_layer.plm.embeddings.word_embeddings.weight if hasattr(self.input_layer, 'plm')\
             else self.input_layer.swv.weight
        return self.output_layer(q_out, o_out, batch, word_embed)
