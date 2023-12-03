# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import yaml
import numpy as np
import torch
from tensorboardX import SummaryWriter

from tool.eval_utils.eval_utils import *
from my_model_name.utils.train_utils import build_dataloader,initLogging
from my_model_name.models import model as model_utils
from my_model_name.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--cfg_file', type=str, default="/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/tool/cfgs/waymo/mtr+100_percent_data.yaml", help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=12, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=16, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default='/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/output/Exp1/ckpt/checkpoint_epoch_29.pth', help='checkpoint to start from')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--save_to_file', action='store_true', default=True, help='')

    args = parser.parse_args()

    return args


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, dist_test=False,cfg=None):
    # load checkpoint
    if args.ckpt is not None: 
        it, epoch = model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    else:
        it, epoch = -1, -1
    model.cuda()

    logger.info(f'*************** LOAD MODEL (epoch={epoch}, iter={it}) for EVALUATION *****************')
    # start evaluation
    eval_one_epoch(
        cfg, model, test_loader, epoch, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )

def main():
    args= parse_config()
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)
    
          
    output_dir = f"{cfg['DATA_CONFIG']['ROOT_DIR']}/output/{args.name}"
    log_path = output_dir+'/log'
    os.makedirs(log_path, exist_ok=True)
    ckpt_dir = output_dir + '/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_output_dir = output_dir + '/eval' + '/test'
    os.makedirs(eval_output_dir, exist_ok=True)  




    log_file = eval_output_dir+'/'+ ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger =initLogging(log_file=log_file)


    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)



    if args.fix_random_seed:
        common_utils.set_random_seed(42)


    test_set, test_loader = build_dataloader(
        dataset_cfg=cfg['DATA_CONFIG'],
        batch_size=args.batch_size,
         workers=args.workers, logger=logger, training=False
    )
    model = model_utils.MotionTransformer(config=cfg['MODEL'])
    with torch.no_grad():
        eval_single_ckpt(model, test_loader, args, eval_output_dir, logger,  dist_test=False,cfg=cfg)


if __name__ == '__main__':
    main()