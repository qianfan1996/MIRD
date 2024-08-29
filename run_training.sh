#!/bin/bash

#seed=(0 1 2 3 42 100 101 111 222 250 520 666 888 999 1024 1111 2021 2022 2023)
#for a in ${seed[*]}; do
#CUDA_VISIBLE_DEVICES=0 python run_training_WSA-BERT.py --seed $a --num_epoch 100 \
#--train_batch_size 48 --learning_rate 0.00001
#done

#seed=(0 1 2 3 42 100 101 111 222 250 520 666 888 999 1024 1111 2021 2022 2023)
#for a in ${seed[*]}; do
#CUDA_VISIBLE_DEVICES=0 python run_training_recon.py --seed $a --num_epoch 100 \
#--train_batch_size 48 --learning_rate 0.00001 --lambda1 0.1 --lambda2 10 --lambda3 100
#done

#seed=(0 1 2 3 42 100 101 111 222 250 520 666 888 999 1024 1111 2021 2022 2023)
#for a in ${seed[*]}; do
#CUDA_VISIBLE_DEVICES=0 python run_training_recon_mim.py --seed $a --num_epoch 100 --multimodal_encoder WSA-BERT \
#--train_batch_size 48 --mi_learning_rate 0.01
#done

seed=(0 1 2 3 42 100 101 111 222 250 520 666 888 999 1024 1111 2021 2022 2023)
for a in ${seed[*]}; do
CUDA_VISIBLE_DEVICES=0 python run_training_recon_mim_sl.py --seed $a --num_epoch 100 --multimodal_encoder WSA-BERT \
--train_batch_size 48 --mi_learning_rate 0.001 --split_rate 2
done
