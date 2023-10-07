#coding=utf8
import os

EXP_PATH = 'exp'


def hyperparam_path(args, create=True):
    if not args.fine_tuning:
        if args.read_model_path and args.testing:
            return args.read_model_path
    exp_path = hyperparam_path_slu(args)
    if create and not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_slu(args):
    domains = '+'.join(sorted(set(args.domains))) if args.domains else 'default'
    task = 'task_%s__dataset_%s__domains_%s' % (args.task, args.dataset, domains)
    task += '__ft_%s' % (args.few_shot) if args.fine_tuning else ''
    # encoder params
    exp_path = '%s' % (args.plm)
    exp_path += '__init_%s' % (args.init_method)
    exp_path += '__ont' if args.ontology_encoding else ''
    exp_path += '__val' if args.use_value else ''
    exp_path += '__enc_%s' % (args.encode_method)
    exp_path += '__hs_%s_x_%s' % (args.hidden_size, args.encoder_num_layers)
    exp_path += '__ca_%s' % (args.cross_attention)
    # exp_path += '__hd_%s' % (args.num_heads)
    # exp_path += '__dp_%s' % (args.dropout)

    # decoder params
    exp_path += '__dec_%s' % (args.decode_method)
    if args.decode_method in ['lf', 'plf']:
        exp_path += '__nl_%s' % (args.decoder_num_layers)

    # training params
    exp_path += '__bs_%s' % (args.batch_size)
    exp_path += '__lr_%s_ld_%s' % (args.lr, args.layerwise_decay)
    exp_path += '__l2_%s' % (args.l2)
    # exp_path += '__wp_%s' % (args.warmup_ratio)
    exp_path += '__sd_%s' % (args.lr_schedule)
    exp_path += '__mi_%s' % (args.max_iter)
    exp_path += '__mn_%s' % (args.max_norm)
    exp_path += '__bm_%s' % (args.beam_size)
    exp_path += '__seed_%s' % (args.seed)
    exp_path = os.path.join(EXP_PATH, task, exp_path)
    return exp_path
