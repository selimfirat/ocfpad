#!/usr/bin/env bash

datas=( replay_attack replay_mobile )
models=( iforest ocsvm ae )
features=( vgg16_faces vgg16_frames vgg16_normalized_faces )
aggregates=( mean max )
normalizess=(" --normalize" "")

for data in "${datas[@]}"
do
    for model in "${models[@]}"
    do
        for feature in "${features[@]}"
        do
            for aggregate in "${aggregates[@]}"
            do
                for normalize in "${normalizess[@]}"
                do
                    python main.py --model $model --data $data --features $feature --aggregate $aggregate $normalize --log results.csv &
                done
            done
        done
    wait
    done
done