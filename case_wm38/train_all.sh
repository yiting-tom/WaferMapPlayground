#!/bin/bash

backbones="mobilenet_v3_large tiny_vit_21m_512"
npz_files="/home/W20862/Projects/WaferMapPlayground/case_wm38/restored_wm38.npz /home/W20862/Projects/WaferMapPlayground/case_wm38/sparse_wm38.npz /home/W20862/Projects/WaferMapPlayground/case_wm38/raw_wm38.npz"
datasizes="38015 1920 19200"
fully_finetune="False True"

# Empty the log file at the start
: > train_all.log

for backbone in $backbones; do
    for npz_file in $npz_files; do
        for datasize in $datasizes; do
            for finetune in $fully_finetune; do
                python ./main.py train \
                    --backbone "$backbone" \
                    --datasize "$datasize" \
                    --npz_file "$npz_file" \
                    --fully_finetune "$finetune" \
                    --lr 0.0001 \
                    --epochs 100 \
                    --resize 224 >> train_all.log 2>&1
            done
        done
    done
done