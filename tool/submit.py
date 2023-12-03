import pickle
from waymo_open_dataset.protos import motion_submission_pb2
import torch
import argparse
import glob
from tqdm import tqdm
import numpy as np
import os
from collections import defaultdict

def parse_config():
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--pred_infos', type=str, default='/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/output/Exp1/eval/test/result.pkl', help='pickle file')
    parser.add_argument('--output_file', type=str, default='/home/zigzagson/Documents/ICRA2023/IV2023_first-Prediction/submit/Valsubmission', help='Output binary file')

    args = parser.parse_args()
    
    return args




def save_submission_to_file(submission, file_path):
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path+'.bin', "wb") as f:
        f.write(submission.SerializeToString())
        

def Submission():
    args = parse_config()
    # Fill up information for submission
    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.account_name = "zhangzhen0509@qq.com"
    submission.unique_method_name = "zigzagson_MTR"
    submission.authors.extend(["zigzagson"])
    submission.affiliation = "NUS"
    submission.description = "zigzagson_MTR100"
    submission.submission_type = (
        motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION  
    )

    pred_dicts = []
    with open(args.pred_infos, 'rb') as f:
        while True:
            try:
                pred_dicts.append(pickle.load(f))
            except EOFError:
                break
    # Initialize a defaultdict of lists
    grouped_dict = defaultdict(list)
    for sub_dicts in pred_dicts:
        for pred_dict in sub_dicts:
    # Append the pred_dict to the list associated with its scenario_id
            grouped_dict[pred_dict['scenario_id']].append(pred_dict)
    for key,value in tqdm(grouped_dict.items()):
        scenario_predictions = motion_submission_pb2.ChallengeScenarioPredictions()  
        scenario_predictions.scenario_id = key        
        for i in range(len(value)):
            prediction_submission = motion_submission_pb2.SingleObjectPrediction()
            prediction_submission.object_id = value[i]['object_id']
            for j in range(value[i]['pred_trajs'].shape[0]):
                scored_trajectory = motion_submission_pb2.ScoredTrajectory() 
                scored_trajectory.confidence = value[i]['pred_scores'][j]
                prediction = value[i]['pred_trajs'][j][4::5,:]
                scored_trajectory.trajectory.center_x.extend(prediction[:,0])
                scored_trajectory.trajectory.center_y.extend(prediction[:,1]) 
                prediction_submission.trajectories.append(scored_trajectory)
            scenario_predictions.single_predictions.predictions.append(prediction_submission)  
        submission.scenario_predictions.append(scenario_predictions)  
          
    save_submission_to_file(submission, args.output_file)
            
    
if __name__ == "__main__":
    
    # Run
    Submission()