
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

from utils.sumo_utils import read_tripInfo, sumo_run
from utils.interface_for_FL import generate_FLtable_from_tripInfo
from utils.options import args_parser

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

    car_num_list = [len(v) for k,v in FL_table.items()]
    L = len(car_num_list)
    car_num_list = car_num_list[int(0.1*L):int(0.9*L)]
    MEAN, VAR = np.mean(car_num_list), np.var(car_num_list)
    print("MEAN", MEAN, "VAR", VAR)

    car_num_dict = {}
    _cnt = 0
    _i = 0
    while _cnt<len(car_num_list):
        car_num_dict[_i] = 0
        for car_num in car_num_list:
            if car_num==_i:
                car_num_dict[_i] += 1
                _cnt += 1
        car_num_dict[_i] /= len(car_num_list)
        _i += 1
    print(car_num_dict.values())


    Lambda = args.Lambda
    T_round = args.round_duration
    H = args.local_iter
    a = args.mu_local_train
    u = 1/args.beta_local_train
    Rc = 420
    v = args.maxSpeed
    T_stay = Rc/v
    D = args.delay_download + args.delay_upload

    tau = np.min([T_round,T_stay])
    p1 = 1/(tau)*(tau-a*H-D-H/u*(1-np.exp(-u/H*(tau-a*H-D))))
    p2 = 1-np.exp(-u/H*(tau-a*H-D))

    E = 2*Lambda*tau*p1 + Lambda*np.abs(T_round-T_stay)*p2
    print('analyzation result:', E)

