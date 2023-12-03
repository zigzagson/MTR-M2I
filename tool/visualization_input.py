import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import time
import matplotlib as mpl
import os 
import pickle
import sys
import math
 
sys.path.append("..")
from my_model_name.datasets.waymo.waymo_types import *
def polygon_completion(polygon):
    polyline_x = []
    polyline_y = []

    for i in range(len(polygon)):
        if i+1 < len(polygon):
            next = i+1
        else:
            next = 0
        dist_x = polygon[next, 0] - polygon[i, 0]
        dist_y = polygon[next, 1] - polygon[i, 1]
        dist = np.linalg.norm([dist_x, dist_y])
        interp_num = np.ceil(dist) * 2
        interp_index = np.arange(2 + interp_num)
        point_x = np.interp(interp_index, [0, interp_index[-1]], [polygon[i, 0], polygon[next, 0]]).tolist()
        point_y = np.interp(interp_index, [0, interp_index[-1]], [polygon[i, 1], polygon[next, 1]]).tolist()
        polyline_x.extend(point_x[:-1])
        polyline_y.extend(point_y[:-1])

    return np.array([polyline_x, polyline_y]).T

def map_viz(map_polyline, map_polyline_mask):
    # (num_topk_polylines, num_points_each_polyline, 9)
    # (num_topk_polylines, num_points_each_polyline)
    num_polylines = map_polyline.shape[0]
    for i in range(num_polylines):
        polyline = map_polyline[i] #(num_points_each_polyline, 9)
        polyline_mask = map_polyline_mask[i] # num_points_each_polyline
        map_type = polyline[0,6]
        polyline = polyline[polyline_mask][:,0:2]
        if map_type == 1 or map_type == 2 or  map_type == 3 :
            plt.plot(polyline[:, 0], polyline[:, 1], 'g', linestyle='solid', linewidth=1)
        elif map_type == 6:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='dashed', linewidth=1)
        elif map_type == 7:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='solid', linewidth=1)
        elif map_type == 8:
            plt.plot(polyline[:, 0], polyline[:, 1], 'w', linestyle='solid', linewidth=1)
        elif map_type == 9:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=1)
        elif map_type == 10:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=1)
        elif map_type == 11:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=1)
        elif map_type == 12:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=1)
        elif map_type == 13:
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:yellow', linestyle='dotted', linewidth=1)
        # else:
        #     plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)
 
        elif map_type == 15:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)
        elif map_type == 16:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)
        elif map_type == 17:
            if len(polyline)<2:
                plt.gca().add_patch(plt.Circle(polyline.reshape(2), 2, color='r'))        
            else:
                for pol in polyline:
                    plt.gca().add_patch(plt.Circle(pol.reshape(2), 2, color='r'))  
                    

        elif map_type == 18:
            polyline = polygon_completion(polyline).astype(np.float32)
            plt.plot(polyline[:, 0], polyline[:, 1], 'b', linewidth=1)
        
        elif map_type == 19:
            polyline = polygon_completion(polyline).astype(np.float32)
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:orange', linewidth=1)

        elif map_type == 20:
            polyline = polygon_completion(polyline).astype(np.float32)
            plt.plot(polyline[:, 0], polyline[:, 1], 'cyan', linewidth=1)
    
        else:
            continue
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)

def agent_viz(obj_traj,obj_traj_mask,obj_traj_future_state,obj_traj_future_mask,ego_index):
    """

    Args:
        obj_traj  (num_objects, num_timestamps=11, num_attrs=29)
        obj_traj_mask (num_objects, num_timestamps=11)
        obj_traj_future_state (num_objects, num_timestamps=80, num_attrs=4) 
        obj_traj_future_mask (_type_): (num_objects, num_timestamps=80) 
    """
    # color ='b'
    # ego_history = obj_traj[ego_index]
    # ego_future = obj_traj_future_state[ego_index]
    # ego_traj_mask  = obj_traj_mask[ego_index]
    # ego_traj_future_mask = obj_traj_future_mask[ego_index]
    # rect = plt.Rectangle((ego_history [-1,0]-ego_history [-1,3]/2, ego_history [-1,1]-ego_history[-1,4]/2), 
    # ego_history[-1,3]+0.5, ego_history[-1,4]+0.5, linewidth=2, color=color, alpha=0.6, zorder=6,
    # transform=mpl.transforms.Affine2D().rotate_around(*(ego_history[-1,0], ego_history[-1,1]), np.arctan2(ego_history[-1,23], ego_history[-1,24])) + plt.gca().transData)
    # plt.gca().add_patch(rect)   
            
    # plt.plot(ego_history[ego_traj_mask][::3, 0], ego_history[ego_traj_mask][::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=4)
    # plt.plot(ego_future[ego_traj_future_mask][::5, 0], ego_future[ego_traj_future_mask][::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)        

    num_objects = obj_traj.shape[0]
    for i in range(num_objects):
        if obj_traj_mask[i,-1]==False:
            continue
        if obj_traj[i,-1,9]==1: 

            color ='r'
            single_traj = obj_traj[i]# (num_timestamps=11, num_attrs=29)
            single_traj_mask = obj_traj_mask[i]
            single_traj_future = obj_traj_future_state[i]
            single_traj_future_mask = (obj_traj_future_mask[i]==1)
            rect = plt.Rectangle((single_traj[-1,0]-single_traj[-1,3]/2, single_traj[-1,1]-single_traj[-1,4]/2), 
            single_traj[-1,3], single_traj[-1,4], linewidth=2, color=color, alpha=0.6, zorder=6,
            transform=mpl.transforms.Affine2D().rotate_around(*(single_traj[-1,0], single_traj[-1,1]), np.arctan2(single_traj[-1,23], single_traj[-1,24])) + plt.gca().transData)
            plt.gca().add_patch(rect)   
                 
            plt.plot(single_traj[single_traj_mask][::3, 0], single_traj[single_traj_mask][::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=5)
            plt.plot(single_traj_future[single_traj_future_mask][::5, 0], single_traj_future[single_traj_future_mask][::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)        

        if obj_traj[i,0,6]==1 and obj_traj[i,0,9]!=1: # vehicle 
            color ='m'
            single_traj = obj_traj[i]# (num_timestamps=11, num_attrs=29)
            single_traj_mask = obj_traj_mask[i]
            single_traj_future = obj_traj_future_state[i]
            single_traj_future_mask = (obj_traj_future_mask[i]==1)
            rect = plt.Rectangle((single_traj[-1,0]-single_traj[-1,3]/2, single_traj[-1,1]-single_traj[-1,4]/2), 
                    single_traj[-1,3], single_traj[-1,4], linewidth=2, color=color, alpha=0.6, zorder=5,
                    transform=mpl.transforms.Affine2D().rotate_around(*(single_traj[-1,0], single_traj[-1,1]), np.arctan2(single_traj[-1,23], single_traj[-1,24])) + plt.gca().transData)
            plt.gca().add_patch(rect)       
            plt.plot(single_traj[single_traj_mask][::3, 0], single_traj[single_traj_mask][::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=4)
            plt.plot(single_traj_future[single_traj_future_mask][::5, 0], single_traj_future[single_traj_future_mask][::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)        

        if obj_traj[i,0,7]==1 and obj_traj[i,0,9]!=1: # pedestrain
            color ='b'
            single_traj = obj_traj[i]# (num_timestamps=11, num_attrs=29)
            single_traj_mask = obj_traj_mask[i]
            single_traj_future = obj_traj_future_state[i]
            single_traj_future_mask =(obj_traj_future_mask[i]==1)
            rect = plt.Rectangle((single_traj[-1,0]-single_traj[-1,3]/2, single_traj[-1,1]-single_traj[-1,4]/2), 
                    single_traj[-1,3], single_traj[-1,4], linewidth=2, color=color, alpha=0.6, zorder=5,
                    transform=mpl.transforms.Affine2D().rotate_around(*(single_traj[-1,0], single_traj[-1,1]), np.arctan2(single_traj[-1,23], single_traj[-1,24])) + plt.gca().transData)           
            plt.gca().add_patch(rect)
            plt.plot(single_traj[single_traj_mask][::3, 0], single_traj[single_traj_mask][::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=4)
            plt.plot(single_traj_future[single_traj_future_mask][::5, 0], single_traj_future[single_traj_future_mask][::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)        

        if obj_traj[i,0,8]==1 and obj_traj[i,0,9]!=1: # cyclist
            color ='g'
            single_traj = obj_traj[i]# (num_timestamps=11, num_attrs=29)
            single_traj_mask = obj_traj_mask[i]
            single_traj_future = obj_traj_future_state[i]
            single_traj_future_mask =(obj_traj_future_mask[i]==1)
            rect = plt.Rectangle((single_traj[-1,0]-single_traj[-1,3]/2, single_traj[-1,1]-single_traj[-1,4]/2), 
                    single_traj[-1,3], single_traj[-1,4], linewidth=2, color=color, alpha=0.6, zorder=5,
                    transform=mpl.transforms.Affine2D().rotate_around(*(single_traj[-1,0], single_traj[-1,1]), np.arctan2(single_traj[-1,23], single_traj[-1,24])) + plt.gca().transData)
            plt.gca().add_patch(rect)
            plt.plot(single_traj[single_traj_mask][::3, 0], single_traj[single_traj_mask][::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=4)
            plt.plot(single_traj_future[single_traj_future_mask][::5, 0], single_traj_future[single_traj_future_mask][::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)        




if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--test_set',default='/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/simulation/processed_scenarios_training' ,type=str, help='path to testing datasets')
    args = parser.parse_args()
    files = glob.glob(args.test_set+'/*')
    for file in files:
        with open(file,'rb') as f:
            obj  = pickle.load(f)
        print(obj.keys())
        time.sleep(100)   
        scenario_ids  = obj['scenario_id'] # [ scenario_id * len(track_index_to_predict)]
        obj_trajs = obj['obj_trajs'] # (num_center_objects, num_objects, num_timestamps=11, num_attrs=29)
        obj_trajs_mask = obj['obj_trajs_mask'] # (num_center_objects, num_objects, num_timestamps=11)
        obj_trajs_future_state = obj['obj_trajs_future_state'] # (num_center_objects, num_objects, num_timestamps=80, num_attrs=4) 
        obj_trajs_future_mask = obj['obj_trajs_future_mask']   # (num_center_objects, num_objects, num_timestamps=80) 
        track_index_to_predict = obj['track_index_to_predict']
        map_polylines = obj['map_polylines']                   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9)
        map_polylines_mask = obj['map_polylines_mask']         # (num_center_objects, num_topk_polylines, num_points_each_polyline)
        assert len(scenario_ids)==obj_trajs.shape[0]== obj_trajs_mask.shape[0]== obj_trajs_future_state.shape[0]==obj_trajs_future_mask.shape[0]==map_polylines.shape[0]==  map_polylines_mask.shape[0]
        num_center_agents = obj_trajs.shape[0]
        for i in range(num_center_agents):
            scenario_id = scenario_ids[i]
            track_index = track_index_to_predict[i]
            obj_traj = obj_trajs[i] # (num_objects, num_timestamps=11, num_attrs=29)
            obj_traj_mask = obj_trajs_mask[i]# (num_objects, num_timestamps=11)
            obj_traj_future_state = obj_trajs_future_state[i] # (num_objects, num_timestamps=80, num_attrs=4) 
            obj_traj_future_mask = obj_trajs_future_mask[i] # (num_objects, num_timestamps=80) 
            map_polyline =map_polylines[i]#(num_topk_polylines, num_points_each_polyline, 9)
            map_polyline_mask = map_polylines_mask[i]#(num_topk_polylines, num_points_each_polyline)
        

            map_viz(map_polyline,map_polyline_mask)
            agent_viz(obj_traj,obj_traj_mask,obj_traj_future_state,obj_traj_future_mask)
            # time.sleep(100)
            plt.gca().set_facecolor('xkcd:grey')
            plt.gca().margins(0)  
            plt.gca().set_aspect('equal')
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.tight_layout()
            save_path = '/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/tool/viz/'

            # Set the filename based on `scenario_id`
            file_name = str(scenario_id) +str(track_index)+ '.png'
            # plt.show()          
            # plt.pause(0.1)  # add a small pause to allow user to interact with the window 

            plt.savefig(save_path + file_name,dpi=300)
            
            # plt.show()          
            # plt.pause(1)  # add a small pause to allow user to interact with the window 
            plt.close()   
                
    