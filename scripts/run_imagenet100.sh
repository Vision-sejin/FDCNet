#!/bin/bash

python main_imagenet.py  \
        --new-classes 10 \
        --start-classes 50 \
        --num_classes 100 \
        --epochs 1 \
        --epochs_task 40 \
        --lr 0.008 \
        --lr_ft 0.001 \
        --save_path 'logs/test0.1' \
        --gpu '0' \
        --w-kd 10 \
        --delta 0.1 \
        --fd 0.001 \
        --fd2 0.05 \
        --kd \
        --K 2000 \
        --save-freq 10 \
        --dataset imagenet100 \
        --root '../ccil/media/data/imagenet100' \
        --batch_size 64 \
        --num-sd 0 \
        --lamda 20 \
        --arch mobilenet \
        --weight_decay 1e-4 \
        --haalland 0.1 \