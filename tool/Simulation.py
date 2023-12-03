import torch
import sys
import csv
import time
import yaml
import argparse
import logging
import os 
os.environ['CUDA_VISIBLE_DEVICE']='0'
import numpy as np
from tqdm import tqdm
from my_model_name.utils.sim_utils import *
from torch.utils.tensorboard import SummaryWriter
from my_model_name.models import model as model_utils
from datetime import datetime
from visualization_input import *
from my_model_name.datasets.waymo.waymo_dataset import WaymoDataset 
from torch.utils.data import DataLoader




def Simulation():
    #################################Initilization############################
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"{cfg['DATA_CONFIG']['ROOT_DIR']}/output/{args.name}/simulation" # '/home/arclab/IV2023_first/output/Exp3'
    log_path = output_dir+'/log'
    os.makedirs(log_path, exist_ok=True)    
    video_path = output_dir + '/video'
    os.makedirs(video_path,exist_ok=True)
    logger =initLogging(log_file=log_path+f"/{current_date}"+'train.log')
    logger.info("------------- {} -------------".format(args.name))
    set_seed(args.seed)
    preprocessed_valid_files = glob.glob(args.preprocessed_valid_set+'/*')  
    model = model_utils.MotionTransformer(config=cfg['MODEL'])
    model.cuda()    
    _,_ = model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.eval()
    total=0
    for valid_file in preprocessed_valid_files:
        info = np.load(valid_file,allow_pickle=True)
        simulator = Simulator(90,cfg['DATA_CONFIG'])
        simulator.load_scenario(info)
        logging.info(f'Scenario: {simulator.scenario_id}')
        batch_dict = simulator.reset()
        done =False
        while not done:
            logging.info(f'Time: {simulator.timestep}')

            # predict
            with torch.no_grad():
                batch_pred_dicts = model(batch_dict)
                final_pred_dicts = generate_prediction_dicts(batch_pred_dicts,K=args.topKPre)
            # plan
            
            # render            
            if args.render:
                simulator.render(final_pred_dicts)            
            
            # take one step
            batch_dict,done = simulator.step()
                

        if args.save:
            simulator.save_animation(video_path) 
        total = total+1
        if total>150:
            break           
        
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--preprocessed_valid_set', type=str, default='/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/simulation/processed_scenarios_validation',help='path to train datasets')
    parser.add_argument('--seed', type=int, help='fix random seed', default=42)   
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers used for dataloader")
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=1)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--ckpt', type=str, default="/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/output/Exp1/ckpt/best_model.pth", help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    # parser.add_argument('--ckpt', type=str, default="/home/arclab/IV2023_first/output/Exp4/ckpt/best_model.pth", help='checkpoint to start from')
    
    parser.add_argument('--cfg_file', type=str, default="/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/tool/cfgs/waymo/simulation.yaml", help='specify the config for training')
    
    parser.add_argument('--render', type=bool, default=True, help='render the figure')
    parser.add_argument('--save', type=bool, default=True, help='save the video')
    parser.add_argument('--topKPre',type=int,default=2,help='The topk predicted trajs to use')


    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    # # Run
    Simulation()