import glob
# import copy
# import random
# import time
import os
import argparse
import tensorflow as tf
# from matplotlib.patches import Polygon as pltPolygon
import pickle
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
from waymo_open_dataset.protos import scenario_pb2


from multiprocessing import Pool
from my_model_name.utils.data_utils import *
from my_model_name.datasets.waymo.waymo_types import *

# import math
# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')

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
                cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 8)
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
            self.polylines = np.concatenate(arrays_to_concatenate).astype(np.float32) 
        except:
            self.polylines = np.zeros((0, 8), dtype=np.float32)
            print('Empty polylines: ')

        


    def decode_tracks_from_proto(self,tracks):
        self.track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': [],
            'track_index':[]
        }
        for track_index,cur_data in enumerate(tracks):  # number of objects
            cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, wrap_to_pi(x.heading),
                                x.velocity_x, x.velocity_y, x.valid,cur_data.object_type], dtype=np.float32) for x in cur_data.states]
            cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp=91, 10)
            self.track_infos['object_id'].append(cur_data.id)
            self.track_infos['object_type'].append(object_type[cur_data.object_type])
            self.track_infos['trajs'].append(cur_traj)
            self.track_infos['track_index'].append(track_index)
        self.track_infos['trajs'] = np.stack(self.track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 10)
      
    def process_data(self, save_path, viz=False):  
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
                # info={
                #     'filter_info':[],
                #     'interesting_index':[]
                # }
                info={}
                info['scenario_id'] = scenario_id   
                ret_info.append(scenario_id)
                
                info['all_polylines'] = self.polylines[:,:-1]        ## (N, 8)
                info['all_agent'] = self.track_infos['trajs'] # (num_objects, num_timestamp, 10)
                info['sdc_track_index'] = sdc_id
                info['object_type'] = self.track_infos['object_type']
                info['object_id'] = self.track_infos['object_id']
                info['predict_list'] = track_to_predict        
                info['dynamic_info'] = self.traffic_light_info
                
                # info['lane'] = self.lane_polylines
                # info['crosswalk'] = self.crosswalk_polylines
                # info['speed_bump'] = self.speed_bump_polylines
                # info['drive_way'] = self.drive_way_polylines
                # info['stop_sign'] = self.stop_signs_polylines
                # info['road_polylines'] =self.road_polylines                      
                                                                          
                # output_file = os.path.join(save_path, f'sample_{scenario_id}.pkl')
                # with open(output_file, 'wb') as f:
                #     pickle.dump(info, f)                    
                    

    
                self.pbar.update(1)

            self.pbar.close()
            
        return ret_info


def multiprocessing(data_files):
    processor = DataProcess([data_files]) 
    ret_info=processor.process_data(save_path, viz=True)
    return ret_info

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Processing')
    # parser.add_argument('--load_path',default="/media/arclab/3THD/Waymo1.2/training", type=str, help='path to dataset files')
    # parser.add_argument('--save_path', default="/home/arclab/IV2023_first/data/waymo/processed_scenarios_training",type=str, help='path to save processed data')
    # parser.add_argument('--load_path',default="/media/arclab/3THD/Waymo1.2/training", type=str, help='path to dataset files')
    # parser.add_argument('--save_path', default="/home/arclab/IV2023_first/data/waymo/processed_scenarios_validation",type=str, help='path to save processed data')
    parser.add_argument('--load_path',default="/media/arclab/3THD/Waymo1.2/training", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/simulation/processed_scenarios_validation",type=str, help='path to save processed data')
    parser.add_argument('--use_multiprocessing', help='if use multiprocessing', default=False)
    
    
    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')
    # num_files = len(data_files)
    # num_selected = int(num_files * 0.2)
    # random.seed(42)  # Set a seed for reproducibility
    # data_files = random.sample(data_files, num_selected)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    if args.use_multiprocessing:
        # print("!!!!!!!!!!!!!!!!!!!!!")
        with Pool() as p:
            results = p.map(multiprocessing, data_files)
            ret_info = [item for sublist in results for item in sublist]
            test_filename = os.path.join('/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/simulation', 'processed_scenarios_test_infos.pkl')
            # test_filename = os.path.join('/home/arclab/IV2023_first/data/waymo', 'processed_scenarios_test_infos.pkl')
            # test_filename = os.path.join('/home/arclab/IV2023_first/data/waymo', 'processed_scenarios_train_infos.pkl')
            # with open(test_filename, 'wb') as f:
            #     pickle.dump(ret_info, f)
    else:
        # print("?????????????????????")
        processor = DataProcess(data_files) 
        processor.process_data(save_path, viz=True)
        print('Done!')
