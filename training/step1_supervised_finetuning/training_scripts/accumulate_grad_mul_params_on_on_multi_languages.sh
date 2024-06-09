

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
total_cards=8

PRETRAIN_OUT=$1
ZERO_STAGE=$2



if [ "$PRETRAIN_OUT" == "" ]; then
    PRETRAIN_OUT=./default_path_to_save
fi


mkdir -p $PRETRAIN_OUT

echo $PRETRAIN_OUT

deepspeed accumulate_grad_mul_params_on_multi_languages.py  \
    --model_name_or_path path_to_model \
    --pretrain_train_data_path path_to_processed_data/train \
    --pretrain_test_data_path path_toprocess_data/test \
    --max_seq_len 512 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --total_cards $total_cards \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --zero_stage 2 \
    --seed 1234 \
    --deepspeed \
    --output_dir $PRETRAIN_OUT \
    &> $PRETRAIN_OUT/training.log
