#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--num_items', type=int, default=1024, help="number of data from every user's local dataset. type: int or list")
    parser.add_argument('--local_train_speed', type=float, default=10., help="the calculation speed of local iteration. local_train_speed * local_train_time = local_iter")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.05, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # model arguments
    parser.add_argument('--model', type=str, default='Lidar2D', help='model name')

    # dataset arguments
    parser.add_argument('--split_ratio', type=float, default=0.9, help="Ratio for splitting original_dataset_train into dataset_train and dataset_validation")
    parser.add_argument('--x_split_num', type=int, default=1, help="Split the training dataset for non-i.i.d. dataset by GPS location. x_split_num means how many splittings we take along the x axis.")
    parser.add_argument('--y_split_num', type=int, default=1, help="Split the training dataset for non-i.i.d. dataset by GPS location. y_split_num means how many splittings we take along the y axis.")
    parser.add_argument('--split_dict', type=int, default=0, help="Specifies which splitting method we will take for non-i.i.d. dataset. 0 means LOS_split_dict, 1 means coord_split_dict.")

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='Raymobtime', help="name of dataset")
    parser.add_argument('--non_iid', action='store_true', default=False,  help='whether i.i.d. or not')
    parser.add_argument('--verbose', action='store_true', default=False,  help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed which make tests reproducible (default: 1)')
    parser.add_argument('--plot_save_path', type=str, default="default", help="The save path for the plots of loss and accuracy.")
    parser.add_argument('--log_save_path', type=str, default="default", help="The save path for the training log of loss and accuracy.")

    # dataset argumrnt by BeamSelection
    parser.add_argument("--gps_data_path", nargs='+', type=str, help="GPS data file")
    parser.add_argument("--lidar_training_data", nargs='+', type=str, help="LIDAR training data file, if you want to merge multiple"
                                                                       " datasets, simply provide a list of paths, as follows:"
                                                                       " --lidar_training_data path_a.npz path_b.npz")
    parser.add_argument("--beam_training_data", nargs='+', type=str, help="Beam training data file, if you want to merge multiple"
                                                                        " datasets, simply provide a list of paths, as follows:"
                                                                        " --beam_training_data path_a.npz path_b.npz")
    parser.add_argument("--lidar_validation_data", nargs='+', type=str, help="LIDAR validation data file, if you want to merge multiple"
                                                                            " datasets, simply provide a list of paths, as follows:"
                                                                            " --lidar_test_data path_a.npz path_b.npz")
    parser.add_argument("--beam_validation_data", nargs='+', type=str, help="Beam validation data file, if you want to merge multiple"
                                                                            " datasets, simply provide a list of paths, as follows:"
                                                                            " --beam_test_data path_a.npz path_b.npz")
    parser.add_argument("--model_path", type=str, default='test_model', help="Path, where the trained model will be saved")
    parser.add_argument("--lidar_test_data", nargs='+', type=str, help="LIDAR test data file, if you want to merge multiple"
                                                                    " datasets, simply provide a list of paths, as follows:"
                                                                    " --lidar_training_data path_a.npz path_b.npz")
    parser.add_argument("--beam_test_data", nargs='+', type=str, default=None,
                        help="Beam test data file, if you want to merge multiple"
                            " datasets, simply provide a list of paths, as follows:"
                            " --beam_training_data path_a.npz path_b.npz")
    parser.add_argument("--preds_csv_path", type=str, default="unnamed_preds.csv",
                        help="Path, where the .csv file with the predictions will be saved")
    
    # SUMO arguments
    parser.add_argument("--no_sumo_run", action="store_true",
                         default=False, help="run sumo simulation to generate tripinfo.xml")
    parser.add_argument("--nogui", action="store_true",
                         default=True, help="run the commandline version of sumo")
    parser.add_argument("--trajectoryInfo_path", type=str,
                         default='./sumo_result/trajectory.csv', help="the file path where stores the trajectory infomation of cars")
    parser.add_argument("--step_length", type=float, 
                         default=0.1, help="sumo sampling interval")
    parser.add_argument("--num_steps", type=int, 
                         default=10000, help="number of time steps, which means how many seconds the car flow takes")
    parser.add_argument("--round_duration", type=float, 
                         default=100, help="duration time of each round")
    parser.add_argument("--beta_download", type=float, 
                         default=1, help="param of shift exponential distribution function for download delay")
    parser.add_argument("--beta_upload", type=float, 
                         default=1, help="param of shift exponential distribution function for upload delay")
    parser.add_argument("--mu_download", type=float, 
                         default=1, help="param of shift exponential distribution function for download delay")
    parser.add_argument("--mu_upload", type=float, 
                         default=1, help="param of shift exponential distribution function for upload delay")
    parser.add_argument("--local_train_time", type=float, 
                         default=5, help="local training time for each vehicle")
    parser.add_argument("--Lambda", type=float, 
                         default=0.1, help="arrival rate of car flow")
    parser.add_argument("--accel", type=float, 
                         default=10, help="accelerate of car flow")
    parser.add_argument("--decel", type=float, 
                         default=20, help="decelerate of car flow")
    parser.add_argument("--sigma", type=float, 
                         default=0, help="imperfection of drivers, which takes value on [0,1], with 0 meaning perfection and 1 meaning imperfection")
    parser.add_argument("--carLength", type=float, 
                         default=5, help="length of cars")
    parser.add_argument("--minGap", type=float, 
                         default=2.5, help="minimum interval between adjacent cars")
    parser.add_argument("--maxSpeed", type=float, 
                         default=20, help="maxSpeed for cars")
    parser.add_argument("--speedFactoer_mean", type=float, 
                         default=1, help="")
    parser.add_argument("--speedFactoer_dev", type=float, 
                         default=0.1, help="")
    parser.add_argument("--speedFactoer_min", type=float, 
                         default=0.5, help="")
    parser.add_argument("--speedFactoer_max", type=float, 
                         default=1.5, help="")
    args = parser.parse_args()
    return args