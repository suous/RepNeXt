#!/usr/bin/env bash

MODEL=repnext_m0
nGPUs=4
epochs=100

torchrun --nproc_per_node=$nGPUs main.py --model $MODEL \
--project cifar-${epochs}e \
--data-set CIFAR \
--data-path ~/.torch \
--batch-size 512 \
--epochs ${epochs} \
--distillation-type none \
--num_workers 16

