task=车载控制_intent_number_3
dataset=aispeech
files='车载控制 车载控制_multi'
domains=''
if [ "$1" = 'aispeech' ] ; then
    #domains='--domains 地图 音乐 天气 打电话 车载控制'
    plm=chinese-electra-180g-small-discriminator
else
    #domains='--domains alarm event messaging music navigation reminder timer weather'
    plm=electra-small-discriminator
fi
seed=999
device=0
ddp='' # --ddp

#plm=chinese-electra-180g-small-discriminator # electra-small-discriminator
encode_method=rgat
decode_method=$2
hidden_size=256
cross_attention=layerwise
encoder_num_layers=2
decoder_num_layers=1
num_heads=8
dropout=0.2

batch_size=20
test_batch_size=50
grad_accumulate=1
eval_after_iter=60
max_iter=100
lr=5e-4
l2=1e-4
layerwise_decay=0.1
lr_schedule=linear
max_norm=5
beam_size=5
n_best=5

python3 -u scripts/train_and_eval.py --task $task --dataset $dataset --files $files $domains --seed $seed --device $device $ddp \
    --ontology_encoding --use_value --init_method swv --encode_method $encode_method --decode_method $decode_method \
    --init_method swv --encode_method $encode_method --decode_method $decode_method \
    --plm $plm --hidden_size $hidden_size --encoder_num_layers $encoder_num_layers --decoder_num_layers $decoder_num_layers \
    --cross_attention $cross_attention --num_heads $num_heads --dropout $dropout \
    --batch_size $batch_size --test_batch_size $test_batch_size --grad_accumulate $grad_accumulate --eval_after_iter $eval_after_iter --max_iter $max_iter \
    --lr $lr --l2 $l2 --layerwise_decay $layerwise_decay --lr_schedule $lr_schedule --max_norm $max_norm --beam_size $beam_size --n_best $n_best
