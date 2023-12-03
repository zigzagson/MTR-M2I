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
from tqdm import tqdm
from my_model_name.utils.sim_utils import *
from my_model_name.models import model as model_utils
from datetime import datetime
from visualization_input import *
from my_model_name.datasets.waymo.waymo_dataset import WaymoDataset 
from torch.utils.data import DataLoader
def get_interested_agents(track_index_to_predict, obj_trajs_full, current_time_index):
    center_objects_list = []
    for k in range(len(track_index_to_predict)):
        obj_idx = track_index_to_predict[k]
        center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
    center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
    return center_objects, np.array(track_index_to_predict)

def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
    """
    Args:
        obj_trajs (num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of obj_trajs
    """
    num_objects, num_timestamps, num_attrs = obj_trajs.shape
    num_center_objects = center_xyz.shape[0]
    assert center_xyz.shape[0] == center_heading.shape[0]
    assert center_xyz.shape[1] in [3, 2]

    obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1) 
    #(num_center_objects,num_objects, num_timestamps, num_attrs)
    obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
        points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
        angle=-center_heading
    ).view(num_center_objects, num_objects, num_timestamps, 2)

    obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

    # rotate direction of velocity
    if rot_vel_index is not None:
        assert len(rot_vel_index) == 2
        obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)
    return obj_trajs

def generate_centered_trajs_for_agents(center_objects, obj_trajs_past, obj_types, center_indices, sdc_index, timestamps, obj_trajs_future):
    """[summary]
    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_trajs_past (num_objects, num_timestamps=11, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_types (num_objects):['TYPE_CYCLIST', 'TYPE_VEHICLE', 'TYPE_VEHICLE',….]
        center_indices (num_center_objects): the index of center objects in obj_trajs_past [index1,index2,...,]
        sdc_index : int
        timestamps ([type]): (11,)
        obj_trajs_future (num_objects, num_future_timestamps=80, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
    Returns:
        ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
        ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
        ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
        ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
    """           
    assert obj_trajs_past.shape[-1] == 10
    assert center_objects.shape[-1] == 10
    num_center_objects = center_objects.shape[0]
    num_objects, num_timestamps, feature_dim = obj_trajs_past.shape
    
    # transform to cpu torch tensor
    center_objects = torch.from_numpy(center_objects).float()
    obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
    timestamps = torch.from_numpy(np.array(10))

    # transform coordinates to the centered objects
    obj_trajs = transform_trajs_to_center_coords(          #(num_center_objects,num_objects, 11, num_attrs)
        obj_trajs=obj_trajs_past,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6, rot_vel_index=[7, 8]
    )
    
    ## generate the attributes for each object
    object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
    object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
    object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  
    object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
    object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
    object_onehot_mask[:, sdc_index, :, 4] = 1
    object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
    object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
    object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps.float()                         # TODO: STRANGE
    object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
    object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
    object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

    vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
    vel_pre = torch.roll(vel, shifts=1, dims=2)
    acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
    acce[:, :, 0, :] = acce[:, :, 1, :]

    ret_obj_trajs = torch.cat((
        obj_trajs[:, :, :, 0:6], 
        object_onehot_mask,
        object_time_embedding, 
        object_heading_embedding,
        obj_trajs[:, :, :, 7:9], 
        acce,
    ), dim=-1)
    #[cx, cy, cz, dx, dy, dz,Whether_Vehicle,Whether_Pedetrain,Whether_Cyclist,Whether_OOI,Whether_SDC,TIME_SEIRES_EMBEDDINGS,COS_theta,SIN_theta,Vx,Xy,ACCEx,ACCEy]
    ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, 11)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs[ret_obj_valid_mask == 0] = 0

    ##  generate label for future trajectories
    obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
    obj_trajs_future = transform_trajs_to_center_coords( # (num_center_obejcts, num_objects, 80,10)
        obj_trajs=obj_trajs_future,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6, rot_vel_index=[7, 8]
    )
    ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
    ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
    ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0
    
    return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()


def generate_centered_traffic_lights_for_agents(center_objects,traffic_lights):
    
    num_traffic_lights, num_timestamps, num_attrs = traffic_lights.shape
    num_center_objects = center_objects.shape[0]
    center_objects = torch.from_numpy(center_objects).float()
    traffic_lights = torch.from_numpy(traffic_lights).float()
    
    
    
    center_xyz = center_objects[:,0:3]
    traffic_lights = traffic_lights.clone().view(1, num_traffic_lights, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
    traffic_lights[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    traffic_lights_data = traffic_lights
    traffic_lights_time_embedding = torch.zeros((num_center_objects, num_traffic_lights, num_timestamps, num_timestamps + 1))
    traffic_lights_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
    traffic_lights_time_embedding[:, :, torch.arange(num_timestamps), -1] = torch.from_numpy( np.array(list(range(10+1)), dtype=np.float32))
    traffic_lights_data = torch.cat((
        traffic_lights[:, :, :, :-1], 
        traffic_lights_time_embedding,
        traffic_lights[:, :, :, -1].unsqueeze(-1), 
    ), dim=-1)        
    
    traffic_lights_data_mask= ~(traffic_lights_data[:, :, :, -1] == 0)
    if (traffic_lights_data[traffic_lights_data_mask].shape[0]==0):
        traffic_lights_data_mask[0,0,0]=True
        traffic_lights_data[0,0,0]=0  
    return traffic_lights_data.numpy(), traffic_lights_data_mask.numpy()>0

def create_agent_data_for_center_objects(
        center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, cur_timestamps,
        obj_types, obj_ids,intertest_agent_filter,traffic_lights
    ):
    """
    Args:
        center_objects : (num_center_objects, num_attrs)
        obj_trajs_past :  (num_objects, 11, 10)
        obj_trajs_future : (num_objects, 80, 10)
        track_index_to_predict : list [index1,index2,...,] int 
        sdc_track_index : int
        timestamps : (11,)
        obj_types : ['TYPE_CYCLIST', 'TYPE_VEHICLE', 'TYPE_VEHICLE',….]
        obj_ids : [agent_id1, agent_id2, agent_id3,... ]
                            
    """
    
    
    obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = generate_centered_trajs_for_agents(
        center_objects=center_objects, obj_trajs_past=obj_trajs_past,
        obj_types=obj_types, center_indices=track_index_to_predict,
        sdc_index=sdc_track_index, timestamps=cur_timestamps, obj_trajs_future=obj_trajs_future
    )

    '''
    obj_trajs_data            # (num_center_objects, num_objects,11,29)
    obj_trajs_mask            # (num_center_objects, num_objects,11)
    obj_trajs_future_state    # (num_center_objects, num_objects,80,4)
    obj_trajs_future_mask     # (num_center_objects, num_objects,80)
    
    '''
    traffic_lights_past = traffic_lights[:,cur_timestamps-10:cur_timestamps+1]
    
    traffic_lights_data,traffic_lights_data_mask = generate_centered_traffic_lights_for_agents(center_objects=center_objects,traffic_lights=traffic_lights_past)    
    # generate the labels of track_objects for training
    center_obj_idxs = np.arange(len(track_index_to_predict))     
    center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]      # (num_center_objects, 80, 4)     
    center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]  # (num_center_objects, 80)
    center_gt_trajs[center_gt_trajs_mask == 0] = 0                                         # (num_center_objects, 80, 4)


    # filter invalid past trajs
    
    assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
    # obj_trajs_past :  (num_objects(original), 11, 10)
    # we will mask the agent if it is not appear in the history
    valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))
    obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
    obj_trajs_data = obj_trajs_data[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
    obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
    obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]  # (num_center_objects, num_objects(filtered), num_timestamps_future):
    obj_types = obj_types[valid_past_mask]  # obj_types : ['TYPE_CYCLIST', 'TYPE_VEHICLE', 'TYPE_VEHICLE',….](filtered)
    obj_ids = obj_ids[valid_past_mask]      # obj_ids : [agent_id1, agent_id2, agent_id3,... ] (filtered)
        
    # after filtering the agents, we need to change the track_to_predict index as well    
    valid_index_cnt = valid_past_mask.cumsum(axis=0)
    track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
    sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

    assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
    assert len(obj_types) == obj_trajs_future_mask.shape[1]
    assert len(obj_ids) == obj_trajs_future_mask.shape[1]

    # generate the final valid position of each object
    obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]                # (num_center_objects, num_objects (filtered), num_timestamps,3)
    num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
    obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32) # num_center_objects, num_objects, 3

    for k in range(num_timestamps):
        cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
        obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask] # for each timestep if the state is valid then update 

    center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
    for k in range(center_gt_trajs_mask.shape[1]):       # center_gt_trajs_mask (num_center_objects, 80)
        cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
        center_gt_final_valid_idx[cur_valid_mask] = k    # the final future timestamp index of the valid state      

    return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
        obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
        obj_types, obj_ids,np.array(track_index_to_predict_new),traffic_lights_data,traffic_lights_data_mask)
def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
    """
    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    point_dim = polylines.shape[-1]
    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[:len(new_polyline)] = new_polyline
        cur_valid_mask[:len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    ret_polylines = torch.from_numpy(ret_polylines)
    ret_polylines_mask = torch.from_numpy(ret_polylines_mask)
    return ret_polylines, ret_polylines_mask

def create_map_data_for_center_objects( center_objects, map_infos, center_offset):
    """
    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        map_infos (dict):
            all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        center_offset (2):, [offset_x, offset_y]
    Returns:
        map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
    """  

    num_center_objects = center_objects.shape[0]

    # transform object coordinates by center objects
    def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
        neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
        neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_objects[:, 6]
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
        neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
            angle=-center_objects[:, 6]
        ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

        # use pre points to map
        # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
        xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
        xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1) # like adding road segment TODO: change the location?
        neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
        return neighboring_polylines, neighboring_polyline_valid_mask

    polylines = torch.from_numpy(map_infos.copy())
    center_objects = torch.from_numpy(center_objects)

    batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
        polylines=polylines.numpy(), point_sampled_interval=1, # todo TODO
        vector_break_dist_thresh=1,
        num_points_each_polyline=20,
    )  
    # batch_polylines.shape (num_polylines, num_points_each_polyline, 7)
    # batch_polylines_mask.shape (num_polylines, num_points_each_polyline)
    
    
    # collect a number of closest polylines for each center objects
    num_of_src_polylines = 786

    if len(batch_polylines) > num_of_src_polylines:
        polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
        center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
        center_offset_rot = common_utils.rotate_points_along_z(
            points=center_offset_rot.view(num_center_objects, 1, 2),
            angle=center_objects[:, 6]
        ).view(num_center_objects, 2)
        
        pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

        dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
        topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
        map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
        map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
    else:
        map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
        map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

    map_polylines, map_polylines_mask = transform_to_center_coordinates(
        neighboring_polylines=map_polylines,
        neighboring_polyline_valid_mask=map_polylines_mask
    )
    # map_polylines [num_center_objects,num_polylines,num_waypoints,feature]
    # map_polylines_mask [num_center_objects,num_polylines,num_waypoints]
    

    temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  
    # (num_center_objects, num_polylines, 3)
    map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  
    # (num_center_objects, num_polylines, 3)

    map_polylines = map_polylines.numpy()#[num_center_objects,num_polylines,num_waypoints,feature]
    map_polylines_mask = map_polylines_mask.numpy() #[num_center_objects,num_polylines,num_waypoints]
    map_polylines_center = map_polylines_center.numpy() #[num_center_objects,num_polylines,3]

        
    return map_polylines, map_polylines_mask, map_polylines_center

def feed_data(cur_timestep,all_agent_trajs,dynamic_info,track_index_to_predict,sdc_track_index,obj_types,
             obj_ids,scenario_id,all_polylines,dataset_cfg ):

    obj_trajs_full = all_agent_trajs
    obj_trajs_past = obj_trajs_full[:, cur_timestep-10:cur_timestep + 1]
    obj_trajs_future = obj_trajs_full[:, cur_timestep + 1:]
    traffic_lights = dynamic_info
    center_objects, track_index_to_predict = get_interested_agents(
        track_index_to_predict=track_index_to_predict,
        obj_trajs_full=obj_trajs_full,
        current_time_index=cur_timestep,
    )      
        
    (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
        center_gt_trajs_mask, center_gt_final_valid_idx, new_obj_types, new_obj_ids,track_index_to_predict_new,traffic_lights_data,traffic_lights_data_mask) = create_agent_data_for_center_objects(
        center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
        track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
        cur_timestamps=cur_timestep, obj_types=obj_types, obj_ids=obj_ids, intertest_agent_filter=track_index_to_predict,traffic_lights = traffic_lights#TODO#self.interesting_index 
    )
    ret_dict = {
        'scenario_id': np.array([scenario_id] * len(track_index_to_predict)),
        'obj_trajs':  torch.from_numpy(obj_trajs_data),
        'obj_trajs_mask':  torch.from_numpy(obj_trajs_mask),
        'obj_trajs_future_state':  torch.from_numpy(obj_trajs_future_state),
        'obj_trajs_future_mask':  torch.from_numpy(obj_trajs_future_mask),            

        'traffic_lights_data':torch.from_numpy(traffic_lights_data),
        'traffic_lights_data_mask':torch.from_numpy(traffic_lights_data_mask),
        
        
        'track_index_to_predict': torch.from_numpy(track_index_to_predict_new),  # used to select center-features
        'origin_track_index_to_predict':torch.from_numpy(track_index_to_predict),
        'obj_trajs_pos':  torch.from_numpy(obj_trajs_pos),
        'obj_trajs_last_pos':  torch.from_numpy(obj_trajs_last_pos),
        'obj_types': obj_types,
        'obj_ids': torch.from_numpy(obj_ids),

        'center_objects_world':  torch.from_numpy(center_objects),
        'center_objects_id': np.array(obj_ids)[track_index_to_predict],
        'center_objects_type': np.array(obj_types)[track_index_to_predict],
        'center_gt_trajs':  torch.from_numpy(center_gt_trajs),
        'center_gt_trajs_mask':  torch.from_numpy(center_gt_trajs_mask),
        'center_gt_final_valid_idx': torch.from_numpy(center_gt_final_valid_idx),
        'center_gt_trajs_src': torch.from_numpy(obj_trajs_full[track_index_to_predict])
    }
    if all_polylines.__len__() == 0:
        all_polylines = np.zeros((2, 7), dtype=np.float32)
        print(f'Warning: empty HDMap {scenario_id}')
    # condition = self.all_polylines[:, -1] != 20  # Create a boolean condition based on the last column
    # filtered_array = self.all_polylines[condition]  # Apply the condition to the array

    map_polylines_data, map_polylines_mask, map_polylines_center = create_map_data_for_center_objects(
        center_objects=center_objects, map_infos=all_polylines,
        center_offset=dataset_cfg['CENTER_OFFSET_OF_MAP']
    )   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)

    ret_dict['map_polylines'] =  torch.from_numpy(map_polylines_data)
    ret_dict['map_polylines_mask'] =  torch.from_numpy((map_polylines_mask > 0))
    ret_dict['map_polylines_center'] =  torch.from_numpy(map_polylines_center)
        
        
    batch_dict={'batch_size':1,'input_dict':ret_dict,'batch_sample_count':[len(ret_dict['track_index_to_predict'])]}    
    return batch_dict

def render(pred_dicts,lane_polylines,road_polylines,stop_signs_polylines,crosswalk_polylines,speed_bump_polylines,drive_way_polylines
           ,all_agent_trajs,track_index_to_predict,sdc_track_index,timestep,dynamic_info,save_path,viz_option,viz_intention=False):
    plt.ion()
    ax = plt.gca()
    fig = plt.gcf()
    dpi = 100
    size_inches = 800 / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_tight_layout(True)
    fig.set_facecolor('xkcd:grey') 
    scenario_id = pred_dicts[0]['scenario_id']

    
    # Viz_map
    
    # for polyline in lane_polylines:
    #     map_type = polyline[0,6]
    #     if map_type == 1 or map_type == 2 or  map_type == 3 :
    #         plt.plot(polyline[:, 0], polyline[:, 1], 'g', linestyle='solid', linewidth=1)
    # viz  roadlines
    for polyline in road_polylines:
        map_type = polyline[0,6]
        if map_type == 6:
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
        elif map_type == 15:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)
        elif map_type == 16:
            plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)
    # viz stop sign
    for polyline in stop_signs_polylines:
        map_type = polyline[0,6]
        if map_type == 17:
            if len(polyline)<2:
                plt.gca().add_patch(plt.Circle(polyline[0][:2], 2, color='r'))        
            else:
                for pol in polyline:
                    plt.gca().add_patch(plt.Circle(pol[0][:2], 2, color='r'))  

    for polyline in crosswalk_polylines:
        map_type = polyline[0,6]
        if map_type == 18:
            polyline = polygon_completion(polyline).astype(np.float32)
            plt.plot(polyline[:, 0], polyline[:, 1], 'b', linewidth=1)
            
    for polyline in speed_bump_polylines:
        map_type = polyline[0,6]
        if map_type == 19:
            polyline = polygon_completion(polyline).astype(np.float32)
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:orange', linewidth=1)

    for polyline in drive_way_polylines:
        map_type = polyline[0,6]
        if map_type == 20:
            polyline = polygon_completion(polyline).astype(np.float32)
            plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:ivory', linewidth=1)



    
    agent_color = ['r', 'm', 'b', 'g'] # [sdc, vehicle, pedestrian, cyclist]
    
    all_agents_track = list(range(len(all_agent_trajs)))
    other_agents_tracks  = [x for x in all_agents_track if x not in track_index_to_predict]
    # sdc
    sdc_current_state = all_agent_trajs[sdc_track_index][timestep]
    color = agent_color[0]
    rect = plt.Rectangle((sdc_current_state[0]-sdc_current_state[3]/2, sdc_current_state[1]-sdc_current_state[4]/2), 
                            sdc_current_state[3], sdc_current_state[4], linewidth=1, color=color, alpha=0.8, zorder=10,
                            transform=mpl.transforms.Affine2D().rotate_around(*(sdc_current_state[0], sdc_current_state[1]), sdc_current_state[6]) + ax.transData)
    ax.add_patch(rect)
    
    plt.plot(all_agent_trajs[sdc_track_index][timestep:-1, 0], all_agent_trajs[sdc_track_index][timestep:-1, 1]\
        , linewidth=1, color=color, marker='*', markersize=2, zorder=20)
    
    # # track_index_to_predict
    # for predict_dict in pred_dicts:
    for i,track_index in enumerate(track_index_to_predict):
        agent_current_state = all_agent_trajs[track_index][timestep]
        agent_all_states = all_agent_trajs[track_index]
        agent_all_states_mask = agent_all_states[:,9]>0
        color = agent_color[int(agent_current_state[-1]) if int(agent_current_state[-1]) in [1, 2] else 3]
        # plt.plot(agent_all_states[agent_all_states_mask][self.timestep:-1, 0], agent_all_states[agent_all_states_mask][self.timestep:-1, 1], linewidth=1, color=color, marker='*', markersize=2, zorder=10)

        rect = plt.Rectangle((agent_current_state[0]-agent_current_state[3]/2, agent_current_state[1]-agent_current_state[4]/2), 
                            agent_current_state[3], agent_current_state[4], linewidth=1, color=color, alpha=0.8, zorder=9,
                            transform=mpl.transforms.Affine2D().rotate_around(*(agent_current_state[0], agent_current_state[1]), agent_current_state[6]) + ax.transData)
        ax.add_patch(rect)  
        
        predict_dict = pred_dicts[i]
        if viz_intention:
            intention_points = predict_dict['intention_points']

            plt.scatter(intention_points[:,0],intention_points[:,1],color='k', marker='o',s=10,zorder=20)
        
        pred_trajs  = predict_dict['pred_trajs']            # [6,80,2]
        pred_scores = predict_dict['pred_scores']           # [6,]
        cmap = cm.get_cmap('viridis')#
        for i in range(len(pred_trajs)):
            pred_color = cmap(pred_scores[i])
            # alpha_value = pred_scores[i]
            zorder_value =int(pred_scores[i] * 10)
            plt.plot(pred_trajs[i][0:80, 0], pred_trajs[i][0:80, 1], linewidth=2, color=pred_color,marker='.', markersize=1, zorder=zorder_value)        
            
            # plt.plot(pred_trajs[i][0:80-(self.timestep-11), 0], pred_trajs[i][0:80-(self.timestep-11), 1], linewidth=2, color=pred_color,marker='.', markersize=1, zorder=zorder_value)        
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(pred_scores), vmax=max(pred_scores)))
    plt.colorbar(sm)         
                    
    for track_index in other_agents_tracks:
        if track_index == sdc_track_index:
            continue
        agent_current_state = all_agent_trajs[track_index][timestep]
        color = 'k'
        rect = plt.Rectangle((agent_current_state[0]-agent_current_state[3]/2, agent_current_state[1]-agent_current_state[4]/2), 
                            agent_current_state[3], agent_current_state[4], linewidth=1, color=color, alpha=0.8, zorder=10,
                            transform=mpl.transforms.Affine2D().rotate_around(*(agent_current_state[0], agent_current_state[1]), agent_current_state[6]) + ax.transData)
        ax.add_patch(rect)             

        
        
    


    # dynamic_map_states
    for signal in dynamic_info[:,timestep]:
        signal_state = signal[-1]
        stop_point = signal[0:2]

        if signal_state in [1, 4, 7]:
            state = 'r' 
        elif signal_state in [2, 5, 8]:
            state = 'y'
        elif signal_state in [3, 6]:
            state = 'g'
        else:
            state = None

        if state:
            light = plt.Circle((stop_point[0], stop_point[1]), 1.2, color=state)
            plt.gca().add_patch(light)

    # show plot
    ax.axis([-100 + sdc_current_state[0], 100 + sdc_current_state[0], -100 + sdc_current_state[1], 100 + sdc_current_state[1]])
    ax.set_aspect('equal')
    ax.grid(False)
    ax.margins(0) 
    ax.axis('off') 
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    # plt.show()
    # plt.pause(0.1)
    if viz_option == 'one_track':
        filename = f"{save_path}/scenario_{scenario_id}_track_{track_index_to_predict}.png"   
    else:
        filename = f"{save_path}/scenario_{scenario_id}.png" 
    plt.savefig(filename)
    plt.clf()
    plt.close()

def Visualization():
    #################################Initilization############################
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"{cfg['DATA_CONFIG']['ROOT_DIR']}/output/{args.name}/simulation" # '/home/arclab/IV2023_first/output/Exp3'
    log_path = output_dir+'/log'
    os.makedirs(log_path, exist_ok=True)    
    picture_path = output_dir + '/picture'
    os.makedirs(picture_path,exist_ok=True)
    logger =initLogging(log_file=log_path+f"/{current_date}"+'train.log')
    logger.info("------------- {} -------------".format(args.name))
    set_seed(args.seed)
    preprocessed_valid_files = glob.glob(args.preprocessed_valid_set+'/*')  
    model = model_utils.MotionTransformer(config=cfg['MODEL'])
    model.cuda()    
    _,_ = model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.eval()
    total = 0
    for valid_file in preprocessed_valid_files:
        info=[]
        info = np.load(valid_file,allow_pickle=True)
        scenario_id            = info['scenario_id']
        sdc_track_index        = info['sdc_track_index']
        obj_types              = info['object_type']
        obj_ids                = info['object_id']
        dynamic_info           = info['dynamic_info']
        all_agent_trajs        = info['all_agent'] 
        lanes_info             = info['lanes_info']
        lane_polylines         = info['lane']           #  list of [n,7] [x,y,z,ori_x,ori_y,ori_z,type]
        road_polylines         = info['road_polylines'] 
        crosswalk_polylines    = info['crosswalk'] 
        speed_bump_polylines   = info['speed_bump'] 
        stop_signs_polylines   = info['stop_sign'] 
        drive_way_polylines    = info['drive_way']
        ref_path               = info['ref_path']
        
        lane_polylines1 = np.concatenate(lane_polylines) if len(lane_polylines)>0 else np.array([])
        road_polylines1 = np.concatenate(road_polylines) if len(road_polylines)>0 else np.array([])
        crosswalk_polylines1 = np.concatenate(crosswalk_polylines) if len(crosswalk_polylines)>0 else np.array([])
        speed_bump_polylines1 = np.concatenate(speed_bump_polylines) if len(speed_bump_polylines)>0 else np.array([])
        stop_signs_polylines1 = np.concatenate(stop_signs_polylines) if len(stop_signs_polylines)>0 else np.array([])
        drive_way_polylines1 = np.concatenate(drive_way_polylines) if len(drive_way_polylines)>0 else np.array([])

        arrays_to_concatenate = []
        if lane_polylines1.size > 0: arrays_to_concatenate.append(lane_polylines1)
        if road_polylines1.size > 0: arrays_to_concatenate.append(road_polylines1)
        if crosswalk_polylines1.size > 0: arrays_to_concatenate.append(crosswalk_polylines1)
        if speed_bump_polylines1.size > 0: arrays_to_concatenate.append(speed_bump_polylines1)
        if stop_signs_polylines1.size > 0: arrays_to_concatenate.append(stop_signs_polylines1)
        if drive_way_polylines1.size > 0: arrays_to_concatenate.append(drive_way_polylines1)
        try:
            all_polylines = np.concatenate(arrays_to_concatenate).astype(np.float32) 
        except:
            all_polylines = np.zeros((2, 7), dtype=np.float32)
            print('Empty polylines: ')
            
        # self.filer_info             = info['filter_info']
        # self.interesting_index      = info['interesting_index']
        # self.track_index_to_predict = list(range(self.all_agent_trajs.shape[0]))   # required to predict
        # if self.sdc_track_index in self.track_index_to_predict:
        #     self.track_index_to_predict.remove(self.sdc_track_index)   # required to predict
        # self.track_index_to_predict.pop(self.sdc_track_index)
        
        tracks_index_to_predict = info['predict_list']
        if args.viz_option == 'one_track':
            for i in range(len(tracks_index_to_predict)):  
                track_index_to_predict = [tracks_index_to_predict[i]] 
                batch_dict = feed_data(cur_timestep=10,all_agent_trajs=all_agent_trajs,dynamic_info=dynamic_info,track_index_to_predict=track_index_to_predict,sdc_track_index=sdc_track_index,
                                    obj_types=obj_types,obj_ids=obj_ids,scenario_id=scenario_id,all_polylines=all_polylines,\
                                        dataset_cfg=cfg['DATA_CONFIG'])
                logging.info(f'Scenario: {scenario_id} at Time:{10}')
                # predict
                with torch.no_grad():
                    batch_pred_dicts = model(batch_dict)
                    final_pred_dicts = generate_prediction_dicts(batch_pred_dicts,K=args.topKPre)         
                render(pred_dicts=final_pred_dicts,lane_polylines=lane_polylines,road_polylines=road_polylines,stop_signs_polylines=stop_signs_polylines,
                    crosswalk_polylines=crosswalk_polylines,speed_bump_polylines=speed_bump_polylines,drive_way_polylines=drive_way_polylines,
                    all_agent_trajs=all_agent_trajs,track_index_to_predict=track_index_to_predict,sdc_track_index=sdc_track_index,
                    timestep=10,dynamic_info=dynamic_info,save_path=picture_path,viz_option=args.viz_option,viz_intention=True)
                total=total+1
                if total>150:
                    break
            if total>150:
                break
        if args.viz_option == 'all_tracks':
            
            track_index_to_predict = tracks_index_to_predict
            batch_dict = feed_data(cur_timestep=10,all_agent_trajs=all_agent_trajs,dynamic_info=dynamic_info,track_index_to_predict=track_index_to_predict,sdc_track_index=sdc_track_index,
                                obj_types=obj_types,obj_ids=obj_ids,scenario_id=scenario_id,all_polylines=all_polylines,\
                                    dataset_cfg=cfg['DATA_CONFIG'])
            logging.info(f'Scenario: {scenario_id} at Time:{10}')
            # predict
            with torch.no_grad():
                batch_pred_dicts = model(batch_dict)
                final_pred_dicts = generate_prediction_dicts(batch_pred_dicts,K=args.topKPre)         
            render(pred_dicts=final_pred_dicts,lane_polylines=lane_polylines,road_polylines=road_polylines,stop_signs_polylines=stop_signs_polylines,
                crosswalk_polylines=crosswalk_polylines,speed_bump_polylines=speed_bump_polylines,drive_way_polylines=drive_way_polylines,
                all_agent_trajs=all_agent_trajs,track_index_to_predict=track_index_to_predict,sdc_track_index=sdc_track_index,
                timestep=10,dynamic_info=dynamic_info,save_path=picture_path,viz_option=args.viz_option,viz_intention=False)
            total=total+1
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
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=2)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    parser.add_argument('--ckpt', type=str, default="/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/output/Exp1/ckpt/best_model.pth", help='checkpoint to start from')
    parser.add_argument('--viz_option',type=str,default='one_track',help='one_track,all_tracks,all_agents')
    # parser.add_argument('--viz_option',type-str,default='one_track',help='one_track,all_tracks,all_agents')
    parser.add_argument('--cfg_file', type=str, default="/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/tool/cfgs/waymo/simulation.yaml", help='specify the config for training')
    parser.add_argument('--topKPre',type=int,default=64,help='The topk predicted trajs to use')


    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    # # Run
    Visualization()