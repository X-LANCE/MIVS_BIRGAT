# coding=utf-8
import logging
import re, math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict


logger = logging.getLogger(__name__)


def set_optimizer(model, args, num_warmup_steps, num_training_steps, last_epoch=-1, module=None, name='plm'):
    module = module if module is not None else model
    use_plm = hasattr(module, name)
    if use_plm and args.layerwise_decay <= 0.: # fix plm params
        for n, p in model.named_parameters():
            if name in n: p.requires_grad = False
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ['bias', 'layer_norm', 'layernorm']
    if use_plm and 0. < args.layerwise_decay <= 0.5: # seperate lr for plm
        grouped_params = [
            {'params': [p for n, p in params if name in n and not any(nd in n.lower() for nd in no_decay)], 'lr': args.layerwise_decay * args.lr, 'weight_decay': args.l2},
            {'params': [p for n, p in params if name in n and any(nd in n.lower() for nd in no_decay)], 'lr': args.layerwise_decay * args.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in params if name not in n and not any(nd in n.lower() for nd in no_decay)], 'weight_decay': args.l2},
            {'params': [p for n, p in params if name not in n and any(nd in n.lower() for nd in no_decay)], 'weight_decay': 0.0},
        ]
        print('Use seperate lr %f for pretrained model ...' % (args.lr * args.layerwise_decay))
    elif use_plm and 0.5 < args.layerwise_decay < 1.: # lr decay layerwise for plm
        pattern = r'encoder\.layer\.(.*?)\.'
        num_layers = int(getattr(module, name).config.num_hidden_layers) if hasattr(getattr(module, name), 'config') else 1
        groups = {"decay": defaultdict(list), "no_decay": defaultdict(list)} # record grouped params
        for n, p in params:
            res = re.search(pattern, n) if name in n else None
            depth = int(res.group(1)) if res is not None else 0 if name in n else num_layers
            if any(nd in n.lower() for nd in no_decay):
                groups["no_decay"][int(depth)].append(p)
            else:
                groups["decay"][int(depth)].append(p)
        grouped_params = []
        for d in groups["decay"]:
            lr = args.lr * (args.layerwise_decay ** (num_layers - d))
            grouped_params.append({'params': groups["decay"][d], 'lr': lr, 'weight_decay': args.l2})
        for d in groups["no_decay"]:
            lr = args.lr * (args.layerwise_decay ** (num_layers - d))
            grouped_params.append({'params': groups["no_decay"][d], 'lr': lr, 'weight_decay': 0.0})
        print('Use layerwise decay (rate %f) lr %f for pretrained model ...' % (args.layerwise_decay, args.lr))
    else: # the same lr for plm and other modules
        grouped_params = [
            {'params': [p for n, p in params if not any(nd in n.lower() for nd in no_decay)], 'weight_decay': args.l2},
            {'params': [p for n, p in params if any(nd in n.lower() for nd in no_decay)], 'weight_decay': 0.0},
        ]
        print('Use the same lr %f for all parameters ...' % (args.lr))
    optimizer = AdamW(grouped_params, lr=args.lr, max_grad_norm=args.max_norm)
    schedule_func = schedule_dict[args.lr_schedule]
    scheduler = schedule_func(optimizer, num_warmup_steps, num_training_steps, last_epoch=last_epoch)
    return optimizer, scheduler


def set_optimizer_bart(model, args, num_warmup_steps, num_training_steps, last_epoch=-1):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ['bias', 'layer_norm', 'layernorm']
    grouped_params = [
        {'params': [p for n, p in params if not any(nd in n.lower() for nd in no_decay)], 'weight_decay': args.l2},
        {'params': [p for n, p in params if any(nd in n.lower() for nd in no_decay)], 'weight_decay': 0.0},
    ]
    print('Use the same lr %f for all parameters ...' % (args.lr))
    optimizer = AdamW(grouped_params, lr=args.lr, max_grad_norm=args.max_norm)
    schedule_func = schedule_dict[args.lr_schedule]
    scheduler = schedule_func(optimizer, num_warmup_steps, num_training_steps, last_epoch=last_epoch)
    return optimizer, scheduler


def get_sqrt_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases according to the formular
    in RATSQL model
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return max(0.0, math.sqrt((num_training_steps - current_step) / float(num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


schedule_dict = {
    "constant": get_constant_schedule,
    "constant_warmup": get_constant_schedule_with_warmup,
    "linear": get_linear_schedule_with_warmup,
    "sqrt": get_sqrt_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
}


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, max_grad_norm=-1, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm, correct_bias=correct_bias)
        super().__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss