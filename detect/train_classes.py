from __future__ import division

import argparse
import os
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--start_index', type=int, required=True)
    parser.add_argument('--end_index', type=int, required=True)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    print(args)

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # setup workdir, num_classes, train anno file per classes index
    sub_dir = '{}-{}'.format(args.start_index, args.end_index)
    cfg.work_dir = os.path.join(cfg.work_dir, sub_dir)
    cfg.data.train.ann_file = cfg.data.train.ann_file + '_' + sub_dir + '.pkl'
    assert isinstance(cfg.model.bbox_head, (list, tuple, dict))

    if isinstance(cfg.model.bbox_head, (list, tuple)):
        for i in range(len(cfg.model.bbox_head)):
            cfg.model.bbox_head[i].num_classes = args.end_index-args.start_index+1
    else:
        cfg.model.bbox_head.num_classes = args.end_index-args.start_index+1
    latest_ckp = os.path.join(cfg.work_dir, 'latest.pth')
    if os.path.exists(latest_ckp):
        cfg.load_from = latest_ckp
    else:
        print('not found: ', latest_ckp)
    print(cfg)
    # patch done


    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = get_dataset(cfg.data.train)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)

def convert_model():
    import torch.nn as nn
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.bbox_head.num_classes = 301
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    print(model)
    print(model.bbox_head.fc_cls.weight.size())
    print(model.bbox_head.fc_cls.bias.size())

    print(cfg.load_from)
    model.load_state_dict(torch.load('./work_dirs/faster_rcnn_r101_fpn_1x/100-400/latest.pth')['state_dict'])
    new_fc = nn.Linear(1024, 101)
    new_fc_reg = nn.Linear(1024, 404)

    new_fc.weight.data[0] = model.bbox_head.fc_cls.weight.data[0]
    new_fc.bias.data[0] = model.bbox_head.fc_cls.bias.data[0]
    #new_fc.weight.data[1:51] = model.bbox_head.fc_cls.weight.data[51:]
    #new_fc.bias.data[1:51] = model.bbox_head.fc_cls.bias.data[51:]
    new_fc.weight.data[1:101] = model.bbox_head.fc_cls.weight.data[201:301]
    new_fc.bias.data[1:101] = model.bbox_head.fc_cls.bias.data[201:301]

    new_fc_reg.weight.data[0:4] = model.bbox_head.fc_reg.weight.data[0:4]
    new_fc_reg.bias.data[0:4] = model.bbox_head.fc_reg.bias.data[0:4]
    #new_fc_reg.weight.data[4:204] = model.bbox_head.fc_reg.weight.data[204:]
    #new_fc_reg.bias.data[4:204] = model.bbox_head.fc_reg.bias.data[204:]
    new_fc_reg.weight.data[4:404] = model.bbox_head.fc_reg.weight.data[804:1204]
    new_fc_reg.bias.data[4:404] = model.bbox_head.fc_reg.bias.data[804:1204]

    model.bbox_head.fc_cls = new_fc
    model.bbox_head.fc_reg = new_fc_reg

    #torch.save(model.state_dict(), './work_dirs/faster_rcnn_r101_fpn_1x/50-100/latest.pth')
    torch.save(model.state_dict(), './work_dirs/faster_rcnn_r101_fpn_1x/300-400/latest.pth')

if __name__ == '__main__':
    main()
    #convert_model()
