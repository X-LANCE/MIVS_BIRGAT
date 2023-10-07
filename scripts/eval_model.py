#coding=utf8
import sys, os, gc, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.example import Example
from utils.batch import Batch
from torch.utils.data import DataLoader


def decode(model, dataset, output_path, batch_size=64, beam_size=5, n_best=5, acc_type='eg-acc', device=None):
    assert acc_type in ['acc', 'beam', 'eg-acc']
    eval_collate_fn = Batch.get_collate_fn(device=device, train=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=eval_collate_fn)
    all_preds, evaluator, decode_method = [], Example.evaluator, Example.decode_method

    model.eval()
    with torch.no_grad():
        for cur_batch in data_loader:
            hyps = model.parse(cur_batch, beam_size=beam_size,  n_best=n_best)
            preds = evaluator.postprocess(hyps, cur_batch.examples, checker=(acc_type == 'eg-acc'), top_ranked=(acc_type != 'beam'), decode_method=decode_method)
            all_preds.extend(preds)

    acc = evaluator.accuracy(all_preds, dataset, output_path)

    torch.cuda.empty_cache()
    gc.collect()
    return acc