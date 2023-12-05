PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py 'configs/bevmap/bevdet-r50.py' 'work_dirs/bevmapdet_reproduce/epoch_24.pth' --eval bbox