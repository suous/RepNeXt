#!/bin/bash

# StarNet's training strategy. Many thanks to the authors for detailed report.
# github: https://github.com/ma-xu/Rewrite-the-Stars
# arxiv:  https://arxiv.org/abs/2111.13566

# Training configuration
# ----------------------
# image_size:      224
# optimizer:       AdamW
# base_lr:         3e-3
# weight_decay:    0.025
# batch_size:      2048
# lr_scheduler:    cosine
# warmup_epochs:   5
# total_epochs:    300
# autoaugment:     rand-m1-mstd0.5-inc1
# label_smoothing: 0.1
# mixup:           0.8
# cutmix:          0.2
# color_jitter:    0.0
# drop_path_rate:  0.0
# amp:             True
# ema:             None
# layer_scale:     None

model=repnext_m1
img_size=224
epochs=300
gpus=4
batch_size=512

torchrun --nproc_per_node=${gpus} moganet_train.py \
--model ${model} \
--experiment ${model}_sz${img_size}_${gpus}xbs${batch_size}_ep${epochs} \
--img_size ${img_size} \
--epochs ${epochs} \
--batch_size ${batch_size} \
--lr 3e-3 \
--weight_decay 0.025 \
--aa rand-m1-mstd0.5-inc1 \
--mixup 0.8 \
--cutmix 0.2 \
--color_jitter 0. \
--amp --native_amp \
--workers 16 \
--data_dir data/imagenet
