PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/train.py configs/bevmap/bevdet-r50.py --work-dir work_dirs/test