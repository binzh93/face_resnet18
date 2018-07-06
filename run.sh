#!/bin/sh
cd /workspace/mnt/group/face-det/zhubin/face_resnet18/
echo '===>Start training!' #>> GenderAge.log
python train.py
echo '===>Training finished!' #>> GenderAge.log
