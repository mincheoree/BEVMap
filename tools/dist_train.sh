#!/usr/bin/env bash

CONFIG='configs/bevdepth/bevdepth-r50.py'
GPUS=2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --work-dir work_dirs/bevdepth_spade2 --resume-from work_dirs/bevdepth_spade2/latest.pth
