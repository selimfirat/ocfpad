#!/usr/bin/env bash

epochs=( 1 )

datas=( replay_attack replay_mobile )
kernels=( 3 )
features=( raw_normalized_faces raw_frames raw_faces)
regs=( 0.01 0.05 0.1)
lrs=( 0.001 0.01 0.1)


for epoch in "${epochs[@]}"
do
    for reg in "${regs[@]}"
    do
        for data in "${datas[@]}"
        do
            for kernel in "${kernels[@]}"
            do
                for lr in "${lrs[@]}"
                do
                    for feature in "${features[@]}"
                    do
                        if [ $data != 'replay_mobile' ] || [ $feature != 'raw_normalized_faces' ]
                        then
                            python convlstm_main.py --data $data --epochs $epoch --kernel_size $kernel --feature $feature --reg $reg --lr $lr --cuda 1 &
                        fi
                    done
                    wait
                done
            done
        done
    done
done