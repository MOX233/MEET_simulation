
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

from utils.sumo_utils import read_tripInfo, sumo_run, sumo_run_with_trajectoryInfo
from utils.interface_for_FL import generate_FLtable_from_tripInfo
from utils.options import args_parser
from utils.FL_training import FL_training

# this is the main entry point of this script
if __name__ == "__main__":
    args = args_parser()
    args.MU_local_train = args.local_iter * args.mu_local_train
    args.BETA_local_train = args.local_iter * args.beta_local_train
    
    if args.no_sumo_run == False:
        os.makedirs(args.sumo_data_dir, exist_ok=True)
        sumo_run(args, save_dir=args.sumo_data_dir)
    car_tripinfo = read_tripInfo(tripInfo_path=os.path.join(args.sumo_data_dir,'tripinfo.xml'))
    FL_table = generate_FLtable_from_tripInfo(args)

    args.gps_data_path = '/home/ubuntu/GraduationProject/code/MEET_simulation/raymobtime_data/Raymobtime_s008/raw_data/CoordVehiclesRxPerScene_s008.csv'
    args.lidar_training_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/lidar_input/lidar_train.npz'
    args.beam_training_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/beam_output/beams_output_train.npz'
    args.lidar_validation_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/lidar_input/lidar_validation.npz'
    args.beam_validation_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/beam_output/beams_output_validation.npz'
    args.lidar_test_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s009/baseline_data/lidar_input/lidar_test.npz'
    args.beam_test_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s009/baseline_data/beam_output/beams_output_test.npz'
    args.model_path = './save/model.pt'

    #args.local_iter = int(args.local_train_speed * args.local_train_time)
    
    FL_training(args,FL_table,car_tripinfo)