#coding=utf8
import sys, os, time, json, gc, itertools, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import Namespace
from contextlib import nullcontext
from utils.args import init_args
from utils.initialization import initialization_wrapper
from utils.example import Example
from utils.batch import Batch
from utils.optimization import set_optimizer, set_optimizer_bart
from model.model_utils import Registrable
from model.model_constructor import *
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from scripts.eval_model import decode

start_time = time.time()

# set environment
args = init_args(sys.argv[1:])
if args.read_model_path: # load from checkpoints or testing mode
    params = json.load(open(os.path.join(args.read_model_path, 'params.json')), object_hook=lambda d: Namespace(**d))
    params.read_model_path, params.lazy_load = args.read_model_path, True
    params.load_optimizer, params.testing, params.device, params.ddp = args.load_optimizer, args.testing, args.device, args.ddp
    params.batch_size, params.grad_accumulate, params.test_batch_size, params.beam_size, params.n_best = args.batch_size, args.grad_accumulate, args.test_batch_size, args.beam_size, args.n_best
    args = params
exp_path, logger, device, local_rank, rank, world_size = initialization_wrapper(args)
is_master = (rank == 0)

# init model
Example.configuration(args.dataset, plm=args.plm, ontology_encoding=args.ontology_encoding, use_value=args.use_value,
    init_method=args.init_method, encode_method=args.encode_method, decode_method=args.decode_method)
model_name = 'encoder-decoder' if not args.encode_method.startswith('gplm') else 'bart-generation'
model = Registrable.by_name(model_name)(args, Example.tranx).to(device)
if args.read_model_path:
    check_point = torch.load(open(os.path.join(args.read_model_path, 'model.bin'), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    logger.info(f"Load saved model from path: {args.read_model_path:s}")
else: json.dump(vars(args), open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
if args.ddp: # add DDP wrapper for model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
base_model = model.module if args.ddp else model


# load dataset
if not args.testing:
    files = args.src_files if args.src_files else args.files
    train_dataset = Example.load_dataset('train', files, args.domains)
    logger.info(f"Dataset size: train -> {len(train_dataset):d} ;")
files = args.tgt_files if args.tgt_files else args.files
dev_dataset = Example.load_dataset('valid', files, args.domains)
logger.info(f"Dataset size: valid -> {len(dev_dataset):d} ;")
logger.info(f"Initialization finished, cost {time.time() - start_time:.4f}s ...")


# training
if not args.testing:
    assert args.batch_size % (world_size * args.grad_accumulate) == 0
    batch_size = args.batch_size // (world_size * args.grad_accumulate)
    # set training dataloader
    train_collate_fn = Batch.get_collate_fn(device=device, train=True)
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, collate_fn=train_collate_fn)
    else: train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=train_collate_fn)
    # set optimizer and scheduler
    eval_per_iter, loss_per_iter = 5000, 1000
    num_training_steps = args.max_iter * loss_per_iter
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    optimizer, scheduler = set_optimizer(base_model, args, num_warmup_steps, num_training_steps, module=base_model.encoder.input_layer, name=args.init_method) if args.init_method != 'gplm' \
        else set_optimizer_bart(base_model, args, num_warmup_steps, num_training_steps)

    iteration, start_epoch, best_result = 0, 0, { 'dev_acc': 0. }
    logger.info(f'Total training steps: {num_training_steps:d};\tWarmup steps: {num_warmup_steps:d}')
    if args.read_model_path and args.load_optimizer:
        optimizer.load_state_dict(check_point['optim'])
        scheduler.load_state_dict(check_point['scheduler'])
        iteration, start_epoch = check_point['iter'], check_point['epoch'] + 1
        best_result = check_point['result']
        logger.info(f'Previous Best Dev Acc is {best_result["dev_acc"]:.4f}')
    logger.info(f'Start training from epoch {start_epoch:d} iteration({loss_per_iter:d}) {iteration // loss_per_iter:d} ......')

    model.train()
    terminate, count, start_time, loss_tracker = False, 0, time.time(), 0
    for i in itertools.count(start_epoch, 1):
        if args.ddp: train_loader.sampler.set_epoch(i)
        for j, current_batch in enumerate(train_loader):
            count += 1
            update_flag = (count == args.grad_accumulate)
            cntx = model.no_sync() if args.ddp and not update_flag else nullcontext()
            with cntx:
                loss = model(current_batch)
                (world_size * loss).backward() # reduction=sum
                loss_tracker += loss.item()
                if update_flag:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    count = 0
                    iteration += 1

                    if iteration % loss_per_iter == 0:
                        logger.info(f'Training iteration({loss_per_iter:d}): {iteration // loss_per_iter:d}\tTime: {time.time() - start_time:.2f}s\tLoss: {loss_tracker:.4f}')
                        start_time, loss_tracker = time.time(), 0
                        torch.cuda.empty_cache()
                        gc.collect()

                    if iteration % eval_per_iter == 0 and iteration < args.eval_after_iter * loss_per_iter and is_master:
                        torch.save({
                            'epoch': i, 'iter': iteration,
                            'model': base_model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'result': best_result
                        }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
                        start_time = time.time()
                    elif iteration % eval_per_iter == 0 and iteration >= args.eval_after_iter * loss_per_iter and is_master:
                        start_time = time.time()
                        dev_acc = decode(base_model, dev_dataset, os.path.join(exp_path, 'dev.iter%s' % (str(iteration // loss_per_iter))), batch_size=args.test_batch_size,
                            beam_size=args.beam_size, n_best=args.n_best, device=device)
                        logger.info(f"Evaluation iteration({loss_per_iter:d}): {iteration // loss_per_iter:d}\tTime: {time.time() - start_time:.2f}s\tDev set acc: {dev_acc:.4f}")
                        if dev_acc >= best_result['dev_acc']:
                            best_result['dev_acc'] = dev_acc
                            torch.save({
                                'epoch': i, 'iter': iteration,
                                'model': base_model.state_dict(),
                                'optim': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'result': best_result
                            }, open(os.path.join(exp_path, 'model.bin'), 'wb'))
                            logger.info(f"NEW BEST MODEL in iteration({loss_per_iter:d}): {iteration // loss_per_iter:d}\tDev set acc: {dev_acc:.4f}")
                        start_time = time.time()
                        model.train()

                    if iteration >= num_training_steps:
                        terminate = True
                        break
        if terminate: break

    if is_master:
        check_point = torch.load(open(os.path.join(exp_path, 'model.bin'), 'rb'), map_location=device)
        del check_point['optim'], check_point['scheduler']
        base_model.load_state_dict(check_point['model'])
        logger.info(f"\nReload saved model in iteration({loss_per_iter:d}) {check_point['iter'] // loss_per_iter:d} from path: {exp_path:s}")


# evaluation
if is_master:
    if args.testing: # Also evaluate on the dev dataset
        start_time = time.time()
        logger.info("Start evaluating on the dev set ......")
        dev_acc = decode(base_model, dev_dataset, os.path.join(exp_path, 'dev.eval'), batch_size=args.test_batch_size,
            beam_size=args.beam_size, n_best=args.n_best, device=device)
        logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s ; Dev set acc: {dev_acc:.4f} ;")

    start_time = time.time()
    test_dataset = Example.load_dataset('test', files, args.domains)
    logger.info(f"Dataset size: test -> {len(test_dataset):d} ;")
    logger.info("Start evaluating on the test set ......")
    start_time = time.time()
    test_acc = decode(base_model, test_dataset, os.path.join(exp_path, 'test.eval'), batch_size=args.test_batch_size,
        beam_size=args.beam_size, n_best=args.n_best, device=device)
    logger.info(f"EVALUATION costs {time.time() - start_time:.2f}s ; Test set acc: {test_acc:.4f} ;")
    check_point['result']['test_acc'] = test_acc
    torch.save(check_point, open(os.path.join(exp_path, 'model.bin'), 'wb'))


if args.ddp:
    dist.destroy_process_group()