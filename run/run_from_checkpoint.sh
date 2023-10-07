#!/bin/bash

ddp=''
read_model_path=exp/task_plm_base__dataset_topv2__domains_alarm+event+messaging+music+navigation+reminder+timer+weather/${1}__init_plm__ont__val__enc_rgat__hs_512_x_2__ca_final__dec_lf__nl_1__bs_20__lr_0.0002_ld_0.8__l2_0.1__sd_linear__mi_100__mn_5.0__bm_5__seed_999
#exp/task_plm_base__dataset_aispeech__domains_地图+天气+打电话+车载控制+音乐/${1}__init_plm__ont__val__enc_rgat__hs_512_x_2__ca_final__dec_lf__nl_1__bs_20__lr_0.0002_ld_0.8__l2_0.1__sd_linear__mi_100__mn_5.0__bm_5__seed_999
batch_size=20
grad_accumulate=2
test_batch_size=20
beam_size=5
n_best=5
device=0


python3 -u scripts/train_and_eval.py --load_optimizer --read_model_path $read_model_path --device $device $ddp \
    --batch_size $batch_size --grad_accumulate $grad_accumulate --test_batch_size $test_batch_size --beam_size $beam_size --n_best $n_best
