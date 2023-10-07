#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    arg_parser = add_argument_encoder(arg_parser)
    arg_parser = add_argument_decoder(arg_parser)
    args = arg_parser.parse_args(params)

    if args.read_model_path: return args # directly return, do not perform argument check
    if args.init_method == 'gplm' or args.encode_method == 'gplm' or args.decode_method.startswith('gplm'):
        args.cross_attention, args.ontology_encoding, args.use_value = 'none', False, False
        args.encoder_num_layers, args.decoder_num_layers = 0, 0
        assert args.init_method == 'gplm' and args.encode_method == 'gplm' and args.decode_method.startswith('gplm'), \
            'Use generative pre-trained language model, the init, encode and decode method must all be "gplm" !'
    if args.decode_method == 'sl+clf':
        assert args.domains is not None, 'If want to use sequence labeling + classifier method, must specify the domains to determine the labels for classifier !'
    if not args.ontology_encoding:
        assert args.domains, 'If want to directly retrieve ontology embeddings, must specify the input domains !'
    if args.encode_method == 'none': args.cross_attention = 'none'
    if args.use_value:
        assert args.ontology_encoding, \
            'If want to use slot value to enhance ontology encoding, please ensure ontology_encoding is set to True !'
        assert args.init_method == 'plm' or args.encode_method != 'none', \
            'If want to use slot values to enhance ontology encoding, please ensure that the init method is plm or the encode method is not none !'
    return args


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--task', default='multi-intent-slu', help='task name')
    arg_parser.add_argument('--dataset', type=str, default='aispeech', choices=['aispeech', 'topv2'])
    arg_parser.add_argument('--files', type=str, nargs='+', help='which files to load')
    arg_parser.add_argument('--domains', type=str, nargs='+', default=None, help='which domains to use, if not specified, use the default domain')
    arg_parser.add_argument('--src_files', type=str, nargs='+', help='source domain files to load')
    arg_parser.add_argument('--tgt_files', type=str, nargs='+', help='target domain files to load')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--ddp', action='store_true', help='use distributed data parallel training')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--read_model_path', type=str, help='read pretrained model path')
    arg_parser.add_argument('--few_shot', default=50, type=int, help='few shot number for fine-tuning')
    arg_parser.add_argument('--fine_tuning', action='store_true', help='fine-tune on few-shot out-of-domain data')
    #### Training Hyperparams ####
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
    arg_parser.add_argument('--test_batch_size', default=64, type=int, help='Test batch size')
    arg_parser.add_argument('--grad_accumulate', default=1, type=int, help='accumulate grad and update once every x steps')
    arg_parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    arg_parser.add_argument('--layerwise_decay', type=float, default=1.0, help='layerwise decay rate for lr, used for PLM')
    arg_parser.add_argument('--l2', type=float, default=1e-4, help='weight decay coefficient')
    arg_parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup steps proportion')
    arg_parser.add_argument('--lr_schedule', default='linear', choices=['constant', 'constant_warmup', 'linear', 'sqrt', 'cosine'], help='lr scheduler')
    arg_parser.add_argument('--eval_after_iter', default=40, type=int, help='Start to evaluate after x 1000 iterations')
    arg_parser.add_argument('--load_optimizer', action='store_true', default=False, help='Whether to load optimizer state')
    arg_parser.add_argument('--max_iter', type=int, default=100, help='number of iterations x 1000')
    arg_parser.add_argument('--max_norm', default=5., type=float, help='clip gradients')
    return arg_parser


def add_argument_encoder(arg_parser):
    # Encoder Hyperparams
    arg_parser.add_argument('--init_method', choices=['plm', 'swv', 'gplm'], default='plm',
        help='using the complete PLM or merely word embedding layer to initialize the input: plm -> the complete auto-encoder PLM ; swv -> static word vectors ; gplm -> BART or T5 generative PLM')
    arg_parser.add_argument('--encode_method', choices=['none', 'gat', 'rgat', 'gplm'], default='rgat',
        help='which encoding method: none -> pure Transformer ; gat/rgat -> use GAT/RGAT model')
    arg_parser.add_argument('--ontology_encoding', action='store_true', help='whether use text description or candidate slot values to encode ontology items')
    arg_parser.add_argument('--plm', type=str, default='chinese-bert-wwm-ext', help='pretrained model name in Huggingface')
    arg_parser.add_argument('--cross_attention', type=str, choices=['layerwise', 'final', 'none'], default='final',
        help='how to adopt cross attention between question words and ontology items: final -> only the last RGAT/GAT layer adopts cross attention')
    arg_parser.add_argument('--use_value', action='store_true', help='whether use slot values to enhance ontology encoding')
    arg_parser.add_argument('--encoder_num_layers', default=8, type=int, help='num of GNN layers in encoder')
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='dimension of hidden states')
    arg_parser.add_argument('--num_heads', default=8, type=int, help='num of heads in multihead attn')
    return arg_parser


def add_argument_decoder(arg_parser):
    # Decoder Hyperparams
    arg_parser.add_argument('--decode_method', choices=['sl', 'lf', 'plf', 'sl+clf', 'gplm', 'gplm+copy'], default='lf', help='method for decoding')
    arg_parser.add_argument('--decoder_num_layers', type=int, default=1, help='num_layers of decoder')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--n_best', default=5, type=int, help='The number of returned hypothesis')
    return arg_parser
