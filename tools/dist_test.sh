#!/usr/bin/env bash

CONFIG='configs/bevmap/bevdet-r50.py'
CHECKPOINT='work_dirs/bevmapdet_reproduce/epoch_24.pth'
GPUS=2
PORT=${PORT:-29600}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
