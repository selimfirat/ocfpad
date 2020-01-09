#!/usr/bin/env bash

python main.py --model iforest --log interdb.csv --normalize --interdb --aggregate mean --features vgg16_frames --data replay_attack
python main.py --model iforest --log interdb.csv --normalize --interdb --aggregate max --features vgg16_faces --data replay_mobile

python main.py --model ocsvm --log interdb.csv --normalize --interdb --aggregate mean --features vgg16_faces --data replay_attack
python main.py --model ocsvm --log interdb.csv --normalize --interdb --aggregate max --features vgg16_faces --data replay_mobile

python main.py --model ae --log interdb.csv --normalize --interdb --aggregate mean --features vgg16_frames --data replay_attack
python main.py --model ae --log interdb.csv --normalize --interdb --aggregate max --features vgg16_faces --data replay_mobile


python convlstm_main.py --interdb --data replay_mobile --epochs 1 --kernel_size 3 --feature raw_frames --reg 0.1 --lr 0.1 --cuda 1