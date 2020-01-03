#!/usr/bin/env bash

python main.py --model iforest --features vgg16_faces --aggregate mean
python main.py --model iforest --features vgg16_faces --aggregate max

python main.py --model iforest --features vgg16_frames --aggregate mean
python main.py --model iforest --features vgg16_frames --aggregate max

python main.py --model iforest --features vggface_faces --aggregate mean
python main.py --model iforest --features vggface_faces --aggregate max

python main.py --model iforest --features vggface_frames --aggregate mean
python main.py --model iforest --features vggface_frames --aggregate max


python main.py --model ocsvm --features vgg16_faces --aggregate mean
python main.py --model ocsvm --features vgg16_faces --aggregate max

python main.py --model ocsvm --features vgg16_frames --aggregate mean
python main.py --model ocsvm --features vgg16_frames --aggregate max

python main.py --model ocsvm --features vggface_faces --aggregate mean
python main.py --model ocsvm --features vggface_faces --aggregate max

python main.py --model ocsvm --features vggface_frames --aggregate mean
python main.py --model ocsvm --features vggface_frames --aggregate max
