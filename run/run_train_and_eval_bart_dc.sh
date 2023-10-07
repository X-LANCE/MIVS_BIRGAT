#!/bin/bash
task=in_domain_combination
dataset=aispeech
src_files='地图 音乐 天气 地图_cross_音乐 音乐_cross_天气 地图_cross_天气'
tgt_files='地图_cross_天气'
domains='地图 音乐 天气'
#domains='alarm event messaging music navigation reminder timer weather'
# files=all_except_null
# domains='地图 音乐 天气 打电话 车载控制'
seed=999
device=0
ddp='' # --ddp

plm=bart-base-chinese #bart-base
decode_method=$1

batch_size=20
test_batch_size=50
grad_accumulate=1
eval_after_iter=60
max_iter=100
lr=5e-5
l2=0.1
lr_schedule=linear
max_norm=5
beam_size=5
n_best=5

python3 -u scripts/train_and_eval.py --task $task --dataset $dataset --src_files $src_files --tgt_files $tgt_files --domains $domains --seed $seed --device $device $ddp \
    --init_method gplm --encode_method gplm --decode_method $decode_method --plm $plm \
    --batch_size $batch_size --test_batch_size $test_batch_size --grad_accumulate $grad_accumulate --eval_after_iter $eval_after_iter --max_iter $max_iter \
    --lr $lr --l2 $l2 --lr_schedule $lr_schedule --max_norm $max_norm --beam_size $beam_size --n_best $n_best
