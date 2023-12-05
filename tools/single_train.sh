PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/train.py configs/bevmap/testdepth.py --work-dir work_dirs/test2
# python tools/train.py configs/bevdepth/bevdepth-r50.py --work-dir work_dirs/test2
