#!/bin/bash

ddp=''
if [ "$1" = "lf" ] ; then
    fp=${1}__nl_1
elif [ "$1" = "plf" ] ; then
    fp=${1}__nl_1
else
    fp=${1}
fi
read_model_path=exp/task_domain_combination__dataset_aispeech__domains_地图+天气+音乐/chinese-electra-180g-small-discriminator__init_swv__ont__val__enc_rgat__hs_256_x_2__ca_layerwise__dec_${fp}__bs_20__lr_0.0005_ld_0.1__l2_0.0001__sd_linear__mi_100__mn_5.0__bm_5__seed_999
#read_model_path=exp/task_车载控制_intent_number_3__dataset_aispeech__domains_车载控制/chinese-bert-wwm-ext__init_plm__ont__val__enc_rgat__hs_512_x_2__ca_final__dec_${fp}__bs_20__lr_0.0002_ld_0.8__l2_0.1__sd_linear__mi_100__mn_5.0__bm_5__seed_999/
few_shot=400
batch_size=20
lr=5e-5
max_iter=10
eval_after_iter=0
grad_accumulate=1
test_batch_size=50
beam_size=5
n_best=5
device=0


python3 -u scripts/train_and_eval_ft.py --fine_tuning --few_shot $few_shot --read_model_path $read_model_path --device $device $ddp \
    --lr $lr --max_iter $max_iter --eval_after_iter $eval_after_iter --batch_size $batch_size --grad_accumulate $grad_accumulate --test_batch_size $test_batch_size --beam_size $beam_size --n_best $n_best
