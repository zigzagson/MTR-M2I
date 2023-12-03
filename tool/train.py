import torch
import sys
import csv
import time
import yaml
import argparse
import logging
import os 
os.environ['CUDA_VISIBLE_DEVICE']='0,1,2,3'
import numpy as np
from torch import nn, optim
from my_model_name.utils.train_utils import *
from torch.utils.tensorboard import SummaryWriter
from my_model_name.models import model as model_utils
from datetime import datetime
from visualization_input import *


def model_training():
    # Logging
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"{cfg['DATA_CONFIG']['ROOT_DIR']}/output/{args.name}" # '/home/arclab/IV2023_first/output/Exp3'
    log_path = output_dir+'/log'
    os.makedirs(log_path, exist_ok=True)                               #'/home/arclab/IV2023_first/output/Exp3/log'
    ckpt_dir = output_dir + '/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)                               #'/home/arclab/IV2023_first/output/Exp3/ckpt'           
    eval_output_dir = output_dir + '/eval' + '/eval_with_train'
    os.makedirs(eval_output_dir, exist_ok=True)                        #'/home/arclab/IV2023_first/output/Exp3/eval/eval_with_train'
    
    logger =initLogging(log_file=log_path+f"/{current_date}"+'train.log')
    logger.info("------------- {} -------------".format(args.name))     
    logger.info("Batch size: {}".format(args.batch_size))
    logger.info("Use device: {}".format(args.device))
    tb_log = SummaryWriter(log_dir=str(output_dir +'/tensorboard'))
    # set seed
    set_seed(args.seed)
    # build dataloader
    train_set,train_loader = build_dataloader(dataset_cfg=cfg['DATA_CONFIG'],
        batch_size=args.batch_size,
        workers=args.num_workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.train_epochs,
        add_worker_init_fn=args.add_worker_init_fn,)
    test_set,test_loader = build_dataloader(dataset_cfg=cfg['DATA_CONFIG'],
        batch_size=args.batch_size,
        workers=args.num_workers,
        logger=logger,
        training=False,)
    
# ########################################################VIZ##########################################################
#     train_loader = iter(train_loader)
#     for batch_dict in train_loader:
#         num_c = batch_dict['input_dict']['obj_trajs'].shape[0]
#         for i in range(num_c):
            
#             scenario_id = batch_dict['input_dict']['scenario_id'][i]
#             ego_index = batch_dict['input_dict']['track_index_to_predict'][i]
#             origin_ego_index =batch_dict['input_dict']['origin_track_index_to_predict'][i].numpy()
#             obj_traj = batch_dict['input_dict']['obj_trajs'][i]
#             obj_traj_mask = batch_dict['input_dict']['obj_trajs_mask'][i]
#             obj_traj_future_state = batch_dict['input_dict']['obj_trajs_future_state'][i] # (num_objects, num_timestamps=80, num_attrs=4) 
#             obj_traj_future_mask = batch_dict['input_dict']['obj_trajs_future_mask'][i] # (num_objects, num_timestamps=80) 
#             map_polyline =batch_dict['input_dict']['map_polylines'][i]#(num_topk_polylines, num_points_each_polyline, 9)
#             map_polyline_mask = batch_dict['input_dict']['map_polylines_mask'][i]#(num_topk_polylines, num_points_each_polyline)
            
#             map_viz(map_polyline,map_polyline_mask)
#             agent_viz(obj_traj,obj_traj_mask,obj_traj_future_state,obj_traj_future_mask,ego_index)
#             # time.sleep(100)
#             plt.gca().set_facecolor('xkcd:grey')
#             plt.gca().margins(0)  
#             plt.gca().set_aspect('equal')
#             plt.gca().axes.get_yaxis().set_visible(False)
#             plt.gca().axes.get_xaxis().set_visible(False) 
#             plt.tight_layout()  
#             save_path = '/home/arclab/IV2023_first/tool/viz/'
#             file_name = str(scenario_id) +str(origin_ego_index)+ '.png'
#             # plt.savefig(save_path + file_name,dpi=300)
#             plt.title(str(scenario_id) +str(origin_ego_index))         
#             plt.show()  
                 
#             plt.close() 
#####################################################################################################################
    model = model_utils.MotionTransformer(config=cfg['MODEL'])
    model.cuda()
    num_total_params = sum([x.numel() for x in model.parameters()])
    logger.info(f'Total number of parameters: {num_total_params}')  
    optimizer = build_optimizer(model, cfg['OPTIMIZATION'])
    for param_group in optimizer.param_groups:
        param_group.setdefault('initial_lr', cfg['OPTIMIZATION']['LR'])    
    scheduler = build_scheduler(
        optimizer, train_loader, cfg['OPTIMIZATION'], total_epochs=args.train_epochs,
        total_iters_each_epoch=len(train_loader), last_epoch=0
    )           
    
    # load checkpoint if it is possible
    
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=False, optimizer=optimizer,
                                                           logger=logger)
        last_epoch = start_epoch + 1
    else:
        it=0
        start_epoch =0
        last_epoch =0

    # -----------------------start training---------------------------
    logger.info('**********************Start training **********************')
    model.train() 
    
    train_model(
        model,
        optimizer,
        train_loader,
        optim_cfg=cfg['OPTIMIZATION'],
        start_epoch=start_epoch,
        total_epochs=args.train_epochs,
        start_iter=it,
        ckpt_save_dir=ckpt_dir,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        tb_log=tb_log,
        scheduler=scheduler,
        logger=logger,
        eval_output_dir=eval_output_dir,
        test_loader=test_loader if not args.not_eval_with_train else None,
        cfg=cfg,  logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval
    )
    logger.info('**********************End training **********************\n\n\n')
    

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp3")
    parser.add_argument('--train_set', type=str, default='/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/simulation/processed_scenarios_training',help='path to train datasets')
    parser.add_argument('--valid_set', type=str, default='/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/simulation/processed_scenarios_validation',help='path to validation datasets')
    parser.add_argument('--seed', type=int, help='fix random seed', default=42)
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers used for dataloader")
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=2)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--ckpt', type=str, default="/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/output/Exp3/ckpt/latest_model.pth", help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    
    parser.add_argument('--pretrain_model_name',type=str, help="the name of the pretrained model")
    parser.add_argument('--cfg_file', type=str, default="/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/tool/cfgs/waymo/mtr+20_percent_data.yaml", help='specify the config for training')
    parser.add_argument('--add_worker_init_fn', action='store_true', default=False, help='')

    
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--logger_iter_interval', type=int, default=100, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--not_eval_with_train', default=False, help='')

    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    # # Run
    model_training()




