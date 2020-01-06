#!/usr/bin/env bash

datas=( replay_attack replay_mobile )
kernels=( 3 5 7 11 21)
epochs=( 1 5 10 )
features=( raw_normalized_faces raw_frames raw_faces)
regs=( 0 0.1 0.25 0.5 1)


for epochs in "${epochs[@]}"
do
    for reg in "${regs[@]}"
    do
        for data in "${datas[@]}"
        do
            for kernel in "${kernels[@]}"
            do
                for feature in features
                do
                    python convlstm_main.py --data $data --epochs $epochs
                done
                wait
            done
        done
    done
done