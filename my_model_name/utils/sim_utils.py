import logging
import torch
import logging
import glob
import random
import numpy as np
from tqdm import tqdm
import os 
import pickle
from my_model_name.utils.data_utils import *
from my_model_name.datasets.waymo.waymo_types import *
from waymo_open_dataset.protos import scenario_pb2
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
import argparse
import matplotlib as mpl
import tensorflow as tf
from multiprocessing import Pool
import matplotlib.animation as animation
from my_model_name.datasets.waymo.waymo_dataset import WaymoDataset
from my_model_name.utils import common_utils
import imageio
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())
    return logging.getLogger()    

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def generate_prediction_dicts( batch_dict,K=1):
    """
    Args:
        batch_dict:
            pred_scores: (num_center_objects, num_modes)
            pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

            input_dict:
            center_objects_world: (num_center_objects, 10)
            center_objects_type: (num_center_objects)
            center_objects_id: (num_center_objects)
            center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
    """
    input_dict = batch_dict['input_dict']

    pred_scores = batch_dict['pred_scores']
    pred_trajs = batch_dict['pred_trajs']
    intention_points = batch_dict['intention_points']
    center_objects_world = input_dict['center_objects_world'].type_as(
        pred_trajs)

    num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
    # assert num_feat == 7

    pred_trajs_world = common_utils.rotate_points_along_z(
        points=pred_trajs.view(
            num_center_objects, num_modes * num_timestamps, num_feat),
        angle=center_objects_world[:, 6].view(num_center_objects)
    ).view(num_center_objects, num_modes, num_timestamps, num_feat)
    pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]
    intention_points = common_utils.rotate_points_along_z(
        points = intention_points,
        angle  = center_objects_world[:, 6].view(num_center_objects)
        
    )
    
    intention_points += center_objects_world[:,None,0:2]
    pred_dict_list = []
    for obj_idx in range(num_center_objects):
        topk_pred_trajs_world = pred_trajs_world[obj_idx, :K, :, 0:2].cpu().numpy()
        tpok_pred_trajs_scores = pred_scores[obj_idx, :K].cpu().numpy()
        single_pred_dict = {
            'scenario_id': input_dict['scenario_id'][obj_idx],
            'pred_trajs': topk_pred_trajs_world,
            'pred_scores': tpok_pred_trajs_scores,
            'track_index_to_predict': input_dict['origin_track_index_to_predict'][obj_idx].cpu().numpy(),
            'intention_points':intention_points[obj_idx].cpu().numpy()
            
        }
        pred_dict_list.append(single_pred_dict)

    return pred_dict_list
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












# Data process
class DataProcess(object):
    def __init__(self, files):
        
        self.data_files = files
        
    def build_map(self, map_features, dynamic_map_states):
        # parsed_data.map_features, parsed_data.dynamic_map_states
        # each scenario has a set of map_features

        self.traffic_light_info={}
        for timestep,dynamic_map_state in enumerate(dynamic_map_states): # num_timestamp
            for cur_signal in dynamic_map_state.lane_states:
                if cur_signal.lane not in self.traffic_light_info:
                    self.traffic_light_info[cur_signal.lane] = np.zeros((91,4))
                    self.traffic_light_info[cur_signal.lane][timestep] = np.array([cur_signal.stop_point.x,cur_signal.stop_point.y,\
                        cur_signal.stop_point.z,cur_signal.state])
                else:
                     self.traffic_light_info[cur_signal.lane][timestep] = np.array([cur_signal.stop_point.x,cur_signal.stop_point.y,\
                        cur_signal.stop_point.z,cur_signal.state])
        
        try:
            self.traffic_light_info = np.stack(list(self.traffic_light_info.values()), axis=0) #[num_of_traffic_lights,T=91,features=4]
        except:
            self.traffic_light_info = np.zeros((1,91,4))
            
        self.lanes={}
        self.lane_polylines=[]
        self.road_polylines=[]
        self.crosswalk_polylines=[]
        self.speed_bump_polylines=[]
        self.stop_signs_polylines=[]
        self.drive_way_polylines=[]
        self.polylines=[]
        # static map features
        for cur_data in map_features:
            map_id = cur_data.id
            if cur_data.lane.ByteSize() > 0:
                self.lanes[map_id] = cur_data.lane
                data_type = lane_type[cur_data.lane.type]
                global_type = polyline_type[data_type]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0) 
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])    # direction vector
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                self.lane_polylines.append(cur_polyline)

                
            elif cur_data.road_line.ByteSize() > 0 :
                data_type = road_line_type[cur_data.road_line.type]
                global_type = polyline_type[data_type]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0) 
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])  
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                self.road_polylines.append(cur_polyline)
       
                
                
            elif cur_data.road_edge.ByteSize() > 0 :
                data_type = road_edge_type[cur_data.road_edge.type]
                global_type = polyline_type[data_type]                
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0) 
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])   
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                self.road_polylines.append(cur_polyline)
                
                                 
            elif cur_data.stop_sign.ByteSize() > 0:
                point = cur_data.stop_sign.position
                global_type = polyline_type['TYPE_STOP_SIGN']
                cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)
                self.stop_signs_polylines.append(cur_polyline)
                
            elif cur_data.crosswalk.ByteSize() > 0:
                global_type = polyline_type['TYPE_CROSSWALK']
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)            
                
                self.crosswalk_polylines.append(cur_polyline)
                
                
            elif cur_data.speed_bump.ByteSize() > 0:
                global_type = polyline_type['TYPE_SPEED_BUMP']
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)            
                self.speed_bump_polylines.append(cur_polyline)
                
                
            elif cur_data.driveway.ByteSize() > 0:    
                global_type = polyline_type['TYPE_DRIVEWAY']
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.driveway.polygon], axis=0)
                cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)      
                self.drive_way_polylines.append(cur_polyline)     
            else:
                raise TypeError  
              
    def decode_tracks_from_proto(self,tracks):
        self.track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': [],
            'track_index':[]
        }
        for track_index,cur_data in enumerate(tracks):  # number of objects
            cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
                                x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
            cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp=91, 10)
            self.track_infos['object_id'].append(cur_data.id)
            self.track_infos['object_type'].append(object_type[cur_data.object_type])
            self.track_infos['trajs'].append(cur_traj)
            self.track_infos['track_index'].append(track_index)
        self.track_infos['trajs'] = np.stack(self.track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 10)
      
    def process_data(self, save_path):  
        ret_info = []  
        for data_file in self.data_files:
             
            dataset = tf.data.TFRecordDataset(data_file)
            self.pbar = tqdm(total=len(list(dataset)))
            self.pbar.set_description(f"Processing {data_file.split('/')[-1]}")

            for data in dataset:
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(data.numpy())   
                scenario_id = parsed_data.scenario_id   #  2fcc1e81956cdf14
                timestamps_seconds = list(parsed_data.timestamps_seconds)
                timestep = parsed_data.current_time_index
                sdc_id = parsed_data.sdc_track_index    # index `for ego vehicle
                track_to_predict = [cur_pred.track_index for cur_pred in parsed_data.tracks_to_predict]
                time_len = len(parsed_data.tracks[sdc_id].states) # time length
                self.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)
                self.decode_tracks_from_proto(parsed_data.tracks)  
                info={
                    # 'filter_info':[],
                    # 'interesting_index':[]
                }
                info['scenario_id'] = scenario_id   
                ret_info.append(scenario_id)      
                # info['all_polylines'] = self.polylines[:,:-1]        ## (N, 8)
                info['all_agent'] = self.track_infos['trajs'] # (num_objects, num_timestamp, 10)
                info['sdc_track_index'] = sdc_id
                info['object_type'] = self.track_infos['object_type']
                info['object_id'] = self.track_infos['object_id']
                info['predict_list'] = track_to_predict
                info['dynamic_info'] = self.traffic_light_info                
                info['lanes_info'] =self.lanes  
                info['lane'] = self.lane_polylines
                info['crosswalk'] = self.crosswalk_polylines
                info['speed_bump'] = self.speed_bump_polylines
                info['drive_way'] = self.drive_way_polylines
                info['stop_sign'] = self.stop_signs_polylines
                info['road_polylines'] =self.road_polylines                
                try:
                    route = find_route(self.track_infos['trajs'][sdc_id], timestep, self.lanes,self.traffic_light_info)
                    ref_path = np.array(route, dtype=np.float32)  
                    if ref_path.shape[0] < 1200:
                        repeated_last_point = np.repeat(ref_path[np.newaxis, -1], 1200-ref_path.shape[0], axis=0)
                        ref_path = np.append(ref_path, repeated_last_point, axis=0)                    
                    info['ref_path'] = ref_path 
                except:
                    continue # filter the rare case
                # for track_index in track_to_predict:
                    
                #     interesting_agents =[]
                #     predict_info={'surrounding_agent_track':[],
                #                   'filtered_agent_track':[]}
                #     predict_info['ego_track_index'] = track_index
                #     interesting_agents.append(track_index)
                    
                #     #################################
                #     ### extract agent information ###
                #     #################################
                #     surrounding_agent = []
                #     filtered_agent =[]                   
                #     total_num = self.track_infos['trajs'].shape[0]
                #     tracks_list = np.arange(0,total_num)
                #     tracks_list =np.setdiff1d(tracks_list,track_index)
                #     ego_agent_trj = self.track_infos['trajs'][track_index].astype(np.float32) 

                #     distance_threshold=3        
                #     for i in tracks_list:  
                #         surrounding_trj = self.track_infos['trajs'][i].astype(np.float32) 
                #         # filtered out the agent that not show up in the past 1 seconds
                #         if sum(self.track_infos['trajs'][i][0:11,9]) == 0: # the agent is not exist in the past
                #             filtered_agent.append(surrounding_trj)
                #             predict_info['filtered_agent_track'].append(i)
                #             continue
                #         # filtered out the agent that is off_road
                #         is_on_road = False
                #         agent_position = Point(surrounding_trj[10][:2])
                #         for map_id,lane_polyline in self.lane_polylines.items():
                #             if lane_polyline.shape[0]==1:
                #                 lane_polyline = np.vstack((lane_polyline, lane_polyline[0]))
                #             # Create a LineString for the lane and check the distance from the agent to the line segment
                #             lane_line = LineString(lane_polyline[:, :2])
                #             distance_to_lane = agent_position.distance(lane_line)
                #             if distance_to_lane <= distance_threshold:
                #                 is_on_road = True
                #                 break 
                        
                #         # claclulate the distance between ego
                #         distance = np.linalg.norm(ego_agent_trj[10,:2]-surrounding_trj[10,:2])
                        
                #         # filter the surrounding agent
                #         if is_agent_visible(ego_agent_trj[0],surrounding_trj[0],self.track_infos['trajs'][:,0,:],15) :
                #             if is_on_road:
                #                 surrounding_agent.append(surrounding_trj) 
                #                 predict_info['surrounding_agent_track'].append(i)
                #                 interesting_agents.append(i)
                #                 continue
                #         if is_agent_visible(ego_agent_trj[10],surrounding_trj[10],self.track_infos['trajs'][:,10,:],150) :
                #             if is_on_road:
                #                 surrounding_agent.append(surrounding_trj) 
                #                 predict_info['surrounding_agent_track'].append(i)
                #                 interesting_agents.append(i)
                #                 continue
                #         if distance<15:
                #             surrounding_agent.append(surrounding_trj)
                #             predict_info['surrounding_agent_track'].append(i)
                #             interesting_agents.append(i)
                             
                #             continue                          
                #         filtered_agent.append(surrounding_trj)
                #         predict_info['filtered_agent_track'].append(i)
                        
                #     if len(surrounding_agent)==0:
                #         for i in tracks_list:
                #             surrounding_trj = self.track_infos['trajs'][i].astype(np.float32)
                #             if i in track_to_predict:
                #                 surrounding_agent.append(surrounding_trj) 
                #                 predict_info['surrounding_agent_track'].append(i)
                #                 interesting_agents.append(i)
                #                 filtered_agent = [trj for trj in filtered_agent if not np.array_equal(trj, surrounding_trj)]
                #                 predict_info['filtered_agent_track'].remove(i)
                                
                #     # surrounding_agent = np.stack(surrounding_agent, axis=0).astype(np.float32) #[N1,91,10]
                #     # filtered_agent = np.stack(filtered_agent, axis=0).astype(np.float32)       #[N2,91,10]
                #     # assert (1+filtered_agent.shape[0]+surrounding_agent.shape[0])== total_num
                #     assert (1+len(predict_info['filtered_agent_track'])+len(predict_info['surrounding_agent_track']))== total_num
                #     info['filter_info'].append(predict_info)
                #     info['interesting_index'].append(interesting_agents)
                    
                output_file = os.path.join(save_path, f'sample_{scenario_id}')
                # with open(output_file, 'wb') as f:
                #     pickle.dump(info, f)                    
                np.savez(output_file, **info)
                self.pbar.update(1)

            self.pbar.close()
            
        return ret_info

def multiprocessing(data_files):
    processor = DataProcess([data_files]) 
    ret_info=processor.process_data(save_path)
    return ret_info

class Simulator(object):
    def __init__(self, timespan,dataset_cfg):
        
        self.timespan = timespan      
        self.dataset_cfg = dataset_cfg
        
    def load_scenario(self, info):
        self.scenario_id            = info['scenario_id']
        self.sdc_track_index        = info['sdc_track_index']
        self.obj_types              = info['object_type']
        self.obj_ids                = info['object_id']
        self.dynamic_info           = info['dynamic_info']
        self.all_agent_trajs        = info['all_agent'] 
        # self.all_polylines          = info['all_polylines']
        self.lanes_info             = info['lanes_info']
        self.lane_polylines         = info['lane']           #  list of [n,7] [x,y,z,ori_x,ori_y,ori_z,type]
        self.road_polylines         = info['road_polylines'] 
        self.crosswalk_polylines    = info['crosswalk'] 
        self.speed_bump_polylines   = info['speed_bump'] 
        self.stop_signs_polylines   = info['stop_sign'] 
        self.drive_way_polylines    = info['drive_way']
        self.ref_path               = info['ref_path']
        
        lane_polylines = np.concatenate(self.lane_polylines) if len(self.lane_polylines)>0 else np.array([])
        road_polylines = np.concatenate(self.road_polylines) if len(self.road_polylines)>0 else np.array([])
        crosswalk_polylines = np.concatenate(self.crosswalk_polylines) if len(self.crosswalk_polylines)>0 else np.array([])
        speed_bump_polylines = np.concatenate(self.speed_bump_polylines) if len(self.speed_bump_polylines)>0 else np.array([])
        stop_signs_polylines = np.concatenate(self.stop_signs_polylines) if len(self.stop_signs_polylines)>0 else np.array([])
        drive_way_polylines = np.concatenate(self.drive_way_polylines) if len(self.drive_way_polylines)>0 else np.array([])

        arrays_to_concatenate = []
        if lane_polylines.size > 0: arrays_to_concatenate.append(lane_polylines)
        if road_polylines.size > 0: arrays_to_concatenate.append(road_polylines)
        if crosswalk_polylines.size > 0: arrays_to_concatenate.append(crosswalk_polylines)
        if speed_bump_polylines.size > 0: arrays_to_concatenate.append(speed_bump_polylines)
        if stop_signs_polylines.size > 0: arrays_to_concatenate.append(stop_signs_polylines)
        if drive_way_polylines.size > 0: arrays_to_concatenate.append(drive_way_polylines)
        try:
            self.all_polylines = np.concatenate(arrays_to_concatenate).astype(np.float32) 
        except:
            self.all_polylines = np.zeros((2, 7), dtype=np.float32)
            print('Empty polylines: ')
            
        # self.filer_info             = info['filter_info']
        # self.interesting_index      = info['interesting_index']
        self.track_index_to_predict = list(range(self.all_agent_trajs.shape[0]))   # required to predict
        if self.sdc_track_index in self.track_index_to_predict:
            self.track_index_to_predict.remove(self.sdc_track_index)   # required to predict
        # self.track_index_to_predict.pop(self.sdc_track_index)
    
    
    def reset(self):
        plt.close('all')
        self.timestep = 10
        self.scene_imgs = []
        batch_dict = self.feed_data(self.timestep)

        return batch_dict

    def step(self):     
        # update timestep 
        if self.timestep<=89:
            self.timestep += 1

            # process data
            batch_dict = self.feed_data(self.timestep)
        
            return batch_dict,False

        # check


        return None, True

    def feed_data(self, cur_timestep, override=False):

        obj_trajs_full = self.all_agent_trajs
        obj_trajs_past = obj_trajs_full[:, cur_timestep-10:cur_timestep + 1]
        obj_trajs_future = obj_trajs_full[:, cur_timestep + 1:]
        traffic_lights = self.dynamic_info
        center_objects, track_index_to_predict = get_interested_agents(
            track_index_to_predict=self.track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=cur_timestep,
        )      
           
        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx, obj_types, obj_ids,track_index_to_predict_new,traffic_lights_data,traffic_lights_data_mask) = create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=self.sdc_track_index,
            cur_timestamps=cur_timestep, obj_types=self.obj_types, obj_ids=self.obj_ids, intertest_agent_filter=track_index_to_predict,traffic_lights = traffic_lights#TODO#self.interesting_index 
        )
        ret_dict = {
            'scenario_id': np.array([self.scenario_id] * len(track_index_to_predict)),
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
            'center_objects_id': np.array(self.obj_ids)[track_index_to_predict],
            'center_objects_type': np.array(self.obj_types)[track_index_to_predict],
            'center_gt_trajs':  torch.from_numpy(center_gt_trajs),
            'center_gt_trajs_mask':  torch.from_numpy(center_gt_trajs_mask),
            'center_gt_final_valid_idx': torch.from_numpy(center_gt_final_valid_idx),
            'center_gt_trajs_src': torch.from_numpy(obj_trajs_full[track_index_to_predict])
        }
        if self.all_polylines.__len__() == 0:
            self.all_polylines = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {self.scenario_id}')
        # condition = self.all_polylines[:, -1] != 20  # Create a boolean condition based on the last column
        # filtered_array = self.all_polylines[condition]  # Apply the condition to the array

        map_polylines_data, map_polylines_mask, map_polylines_center = create_map_data_for_center_objects(
            center_objects=center_objects, map_infos=self.all_polylines,
            center_offset=self.dataset_cfg['CENTER_OFFSET_OF_MAP']
        )   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)

        ret_dict['map_polylines'] =  torch.from_numpy(map_polylines_data)
        ret_dict['map_polylines_mask'] =  torch.from_numpy((map_polylines_mask > 0))
        ret_dict['map_polylines_center'] =  torch.from_numpy(map_polylines_center)
            
            
        batch_dict={'batch_size':1,'input_dict':ret_dict,'batch_sample_count':[len(ret_dict['track_index_to_predict'])]}    
        return batch_dict

    def render(self,pred_dicts):
        plt.ion()
        ax = plt.gca()
        fig = plt.gcf()
        dpi = 100
        size_inches = 800 / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)
        fig.set_facecolor('xkcd:grey') 
        # Viz reference path
        
        # Viz_map
        
        # for key,polyline in self.lane_polylines.item().items():
        #     map_type = polyline[0,6]
        #     if map_type == 1 or map_type == 2 or  map_type == 3 :
        #         plt.plot(polyline[:, 0], polyline[:, 1], 'g', linestyle='solid', linewidth=1)
        # viz  roadlines
        for polyline in self.road_polylines:
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
        for polyline in self.stop_signs_polylines:
            map_type = polyline[0,6]
            if map_type == 17:
                if len(polyline)<2:
                    plt.gca().add_patch(plt.Circle(polyline[0][:2], 2, color='r'))        
                else:
                    for pol in polyline:
                        plt.gca().add_patch(plt.Circle(pol[0][:2], 2, color='r'))  
 
        for polyline in self.crosswalk_polylines:
            map_type = polyline[0,6]
            if map_type == 18:
                polyline = polygon_completion(polyline).astype(np.float32)
                plt.plot(polyline[:, 0], polyline[:, 1], 'b', linewidth=1)
                
        for polyline in self.speed_bump_polylines:
            map_type = polyline[0,6]
            if map_type == 19:
                polyline = polygon_completion(polyline).astype(np.float32)
                plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:orange', linewidth=1)

        for polyline in self.drive_way_polylines:
            map_type = polyline[0,6]
            if map_type == 20:
                polyline = polygon_completion(polyline).astype(np.float32)
                plt.plot(polyline[:, 0], polyline[:, 1], 'xkcd:ivory', linewidth=1)



        
        agent_color = ['r', 'm', 'b', 'g'] # [sdc, vehicle, pedestrian, cyclist]
        
        all_agents_track = list(range(len(self.all_agent_trajs)))
        other_agents_tracks  = [x for x in all_agents_track if x not in self.track_index_to_predict]
        # sdc
        sdc_current_state = self.all_agent_trajs[self.sdc_track_index][self.timestep]
        color = agent_color[0]
        rect = plt.Rectangle((sdc_current_state[0]-sdc_current_state[3]/2, sdc_current_state[1]-sdc_current_state[4]/2), 
                              sdc_current_state[3], sdc_current_state[4], linewidth=1, color=color, alpha=0.8, zorder=10,
                              transform=mpl.transforms.Affine2D().rotate_around(*(sdc_current_state[0], sdc_current_state[1]), sdc_current_state[6]) + ax.transData)
        ax.add_patch(rect)
        
        plt.plot(self.all_agent_trajs[self.sdc_track_index][self.timestep:-1, 0], self.all_agent_trajs[self.sdc_track_index][self.timestep:-1, 1]\
            , linewidth=1, color=color, marker='*', markersize=2, zorder=20)
        
        # # track_index_to_predict
        # for predict_dict in pred_dicts:
        for i,track_index in enumerate(self.track_index_to_predict):
            agent_current_state = self.all_agent_trajs[track_index][self.timestep]
            agent_all_states = self.all_agent_trajs[track_index]
            agent_all_states_mask = agent_all_states[:,9]>0
            color = agent_color[int(agent_current_state[-1]) if int(agent_current_state[-1]) in [1, 2] else 3]
            # plt.plot(agent_all_states[agent_all_states_mask][self.timestep:-1, 0], agent_all_states[agent_all_states_mask][self.timestep:-1, 1], linewidth=1, color=color, marker='*', markersize=2, zorder=10)

            rect = plt.Rectangle((agent_current_state[0]-agent_current_state[3]/2, agent_current_state[1]-agent_current_state[4]/2), 
                                agent_current_state[3], agent_current_state[4], linewidth=1, color=color, alpha=0.8, zorder=9,
                                transform=mpl.transforms.Affine2D().rotate_around(*(agent_current_state[0], agent_current_state[1]), agent_current_state[6]) + ax.transData)
            ax.add_patch(rect)  
            
            predict_dict = pred_dicts[i]
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
            if track_index == self.sdc_track_index:
                continue
            agent_current_state = self.all_agent_trajs[track_index][self.timestep]
            color = 'k'
            rect = plt.Rectangle((agent_current_state[0]-agent_current_state[3]/2, agent_current_state[1]-agent_current_state[4]/2), 
                                agent_current_state[3], agent_current_state[4], linewidth=1, color=color, alpha=0.8, zorder=10,
                                transform=mpl.transforms.Affine2D().rotate_around(*(agent_current_state[0], agent_current_state[1]), agent_current_state[6]) + ax.transData)
            ax.add_patch(rect)             
 
            
            
        
 

        # dynamic_map_states
        for signal in self.dynamic_info[:,self.timestep]:
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
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))   
        self.scene_imgs.append(data)   
        # plt.show()
        plt.pause(0.1)
        plt.clf()


    def save_animation(self, path='.'):
        images = np.array(self.scene_imgs)  # assuming self.scene_imgs is your array of images
        writer = imageio.get_writer(path+'/'+f'{self.scenario_id}.mp4', fps=10)  # change fps as needed

        for img in images:
            writer.append_data(img)

        writer.close()
    


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--load_path',default="/home/Documents/ICRA2023/IV2023_first-Prediction/simulation/validation", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="/home/Documents/ICRA2023/IV2023_first-Prediction/simulation/processed_scenarios_training",type=str, help='path to save processed data')
    parser.add_argument('--use_multiprocessing', help='if use multiprocessing', default=True)
    
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    if args.use_multiprocessing:
        # print("!!!!!!!!!!!!!!!!!!!!!")
        with Pool() as p:
            results = p.map(multiprocessing, data_files)
            ret_info = [item for sublist in results for item in sublist]
            test_filename = os.path.join('/home/Documents/ICRA2023/IV2023_first-Prediction/simulation', 'processed_scenarios_test_infos.pkl')
            # test_filename = os.path.join('/home/arclab/IV2023_first/data/waymo', 'processed_scenarios_train_infos.pkl')
            with open(test_filename, 'wb') as f:
                pickle.dump(ret_info, f)
    else:
        # print("?????????????????????")
        processor = DataProcess(data_files) 
        processor.process_data(save_path)
        print('Done!')
