CONFIG=$1
GPUS=$2

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
PYTHOHPATH=./ python3  detect/train_classes.py --gpus $GPUS $CONFIG ${@:3}
