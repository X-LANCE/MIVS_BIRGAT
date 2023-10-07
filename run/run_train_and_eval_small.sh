task=ablate_encoder
dataset=topv2
files=all
#domains='--domains 地图 音乐 天气 打电话 车载控制'
domains='--domains alarm event messaging music navigation reminder timer weather'
# files=all_except_null
# domains='' # '--domains 地图 音乐 天气 打电话 车载控制'
seed=999
device=0
ddp='' # --ddp

plm=electra-small-discriminator #chinese-electra-180g-small-discriminator
init_method=plm
encode_method=$1
decode_method=lf
hidden_size=256
cross_attention=final
encoder_num_layers=2
decoder_num_layers=1
num_heads=8
dropout=0.2

batch_size=20
test_batch_size=50
grad_accumulate=1
eval_after_iter=60
max_iter=100
lr=4e-4
l2=0.1
layerwise_decay=0.8
lr_schedule=linear
max_norm=5
beam_size=5
n_best=5

python3 -u scripts/train_and_eval.py --task $task --dataset $dataset --files $files $domains --seed $seed --device $device $ddp \
    --use_value --ontology_encoding --init_method plm --encode_method $encode_method --decode_method $decode_method \
    --plm $plm --hidden_size $hidden_size --encoder_num_layers $encoder_num_layers --decoder_num_layers $decoder_num_layers \
    --cross_attention $cross_attention --num_heads $num_heads --dropout $dropout \
    --batch_size $batch_size --test_batch_size $test_batch_size --grad_accumulate $grad_accumulate --eval_after_iter $eval_after_iter --max_iter $max_iter \
    --lr $lr --l2 $l2 --layerwise_decay $layerwise_decay --lr_schedule $lr_schedule --max_norm $max_norm --beam_size $beam_size --n_best $n_best
