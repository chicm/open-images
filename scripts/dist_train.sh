
PYTHONPATH=./ python -m torch.distributed.launch --nproc_per_node=4 \
    ./mmdetection/tools/train.py $1 --launcher pytorch ${@:3}
