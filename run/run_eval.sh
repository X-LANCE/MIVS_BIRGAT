#!/bin/bash

read_model_path=exp/task_车载控制_intent_number_3__dataset_aispeech__domains_车载控制/chinese-electra-180g-small-discriminator__init_swv__enc_rgat__hs_256_x_2__ca_layerwise__dec_lf__nl_1__bs_20__lr_0.0005_ld_0.1__l2_0.0001__sd_linear__mi_100__mn_5.0__bm_5__seed_999/
test_batch_size=20
beam_size=5
n_best=5
device=0


python3 -u scripts/train_and_eval.py --read_model_path $read_model_path \
    --test_batch_size $test_batch_size --beam_size $beam_size --n_best $n_best --device $device --testing
