#!/bin/sh

hydra_training(){
     python main.py  --gpu $1 --exp_name $2 --model_name $3 --train_method $4 \
    --test_method pgd --t $5 --lr $6 --lr_policy $7 \
    --data_name SARS_COV_2 --num_classes 2 --batch_size 64 --input_size 224 --epoch ${8} --tmax 200 \
    --data_dir '/share_data/dataset' --base_dir '/share_data' \
    --optimizer ${9} --num_steps 10  --beta ${10} --epsilon ${11} --step_size ${12}  --distance "l_inf" --es_patience 20 \
    --evl_epsilon ${13} --evl_step_size ${14} --evl_num_steps ${15}  --awp_warmup 20 \
    --clip_score ${16} --gamma ${17} --decay_step 10  --pretrained  1 --aug sgf
}

(
    hydra_training  0   "MMKDRO"  deit_t_distilled  MMKDRO   0.003  0.0001 cosineAnn  200 adamw  3  0.03 0.0784  0.03 0.0784 10  0.1  0.8;

);
