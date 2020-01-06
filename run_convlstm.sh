#!/usr/bin/env bash

epochs=( 1 )

datas=( replay_attack replay_mobile )
kernels=( 3 5 7 11 21)
features=( raw_normalized_faces raw_frames raw_faces)
regs=( 0 0.1 0.25 0.5 1)


for epoch in "${epochs[@]}"
do
    for reg in "${regs[@]}"
    do
        for data in "${datas[@]}"
        do
            for kernel in "${kernels[@]}"
            do
                for feature in "${features[@]}"
                do
                    python convlstm_main.py --data $data --epochs $epoch --kernel_size $kernel --feature $feature --reg $reg --lr 0.001 --cuda 0 &
                done
                wait
            done
        done
    done
done