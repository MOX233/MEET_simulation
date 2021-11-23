#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import random
import numpy as np
import torch
import sys 
sys.path.append("..") 

from torch.utils.data.dataloader import DataLoader

from models.nets import Lidar2D
from utils.dataloader import LidarDataset2D
from utils.options import args_parser
from utils.sampling import Raymobtime_iid, Raymobtime_noniid
from utils.update import LocalUpdate
from utils.federate_learning_avg import FedAvg
from utils.evaluator import test_beam_select
from utils.plot_utils import plot_loss_curve, plot_acc_curve, plot_loss_acc_curve

def FL_training(args,FL_table,car_tripinfo):
    # parse args
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # fix random_seed, so that the experiment can be repeat
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and split users
    num_users = len(car_tripinfo)
    if args.dataset == 'Raymobtime':
        original_dataset_train = LidarDataset2D(lidar_data_path=args.lidar_training_data, beam_data_path=args.beam_training_data)
        dataset_test = LidarDataset2D(lidar_data_path=args.lidar_validation_data, beam_data_path=args.beam_validation_data)
        # split original dataset_train into dataset_train and dataset_val
        dataset_train, dataset_val = original_dataset_train.split(split_ratio=args.split_ratio)

        if args.iid:
            dict_users = Raymobtime_iid(dataset_train, args.num_items, num_users)
        else:
            dict_users = Raymobtime_noniid(dataset_train, num_users)
    else:
        exit('Error: unrecognized dataset')
    
    # build model
    if args.model == 'Lidar2D':
        net_glob = Lidar2D(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    loss_val = []
    acc_val = []
    net_best = net_glob
    rounds = len(FL_table.keys())
    for round in range(rounds):
        loss_locals = []
        w_locals = []
        
        #idxs_users = np.random.choice(range(num_users), m, replace=False)
        idxs_users = [int(car.split('_')[-1]) for car in FL_table[round].keys()]
        if idxs_users == []:

            # print loss
            loss_avg = loss_train[-1] if round>0 else 1
            print('Round {:3d}, No car, Average loss {:.3f}'.format(round, loss_avg), end=' ')
            loss_train.append(loss_avg)
    
            # validation part
            iter_val_loss = loss_val[-1] if round>0 else 1
            [iter_val_top1_acc, iter_val_top5_acc, iter_val_top10_acc] = acc_val[-1] if round>0 else [0. ,0. ,0.]
            loss_val.append(iter_val_loss)
            acc_val.append([iter_val_top1_acc, iter_val_top5_acc, iter_val_top10_acc])
            print("Validation Accuracy: Top-1:{:.4f}% Top-5:{:.4f}% Top-10:{:.4f}%".format(iter_val_top1_acc * 100., iter_val_top5_acc * 100., iter_val_top10_acc * 100.))
            
        else:
            for idx in idxs_users:
                #local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                #w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], local_bs=args.local_bs)
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), local_iter=args.local_iter)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = FedAvg(w_locals)
    
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
    
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(round, loss_avg), end=' ')
            loss_train.append(loss_avg)
    
            # validation part
            [iter_val_top1_acc, iter_val_top5_acc, iter_val_top10_acc], iter_val_loss = test_beam_select(net_glob, dataset_val, args)
            loss_val.append(iter_val_loss)
            acc_val.append([iter_val_top1_acc, iter_val_top5_acc, iter_val_top10_acc])
            print("Validation Accuracy: Top-1:{:.4f}% Top-5:{:.4f}% Top-10:{:.4f}%".format(iter_val_top1_acc * 100., iter_val_top5_acc * 100., iter_val_top10_acc * 100.))
            plot_loss_acc_curve(loss_train, loss_val, acc_val, rounds, args)
    # plot loss and acc curve
    # plot_loss_acc_curve(loss_train, loss_val, acc_val, rounds, args)

    # test part
    net_glob.eval()

    acc_train, loss_train = test_beam_select(net_glob, dataset_train, args) 
    acc_test, loss_test = test_beam_select(net_glob, dataset_test, args)    
    [top1_acc_train, top5_acc_train, top10_acc_train] = acc_train
    [top1_acc_test, top5_acc_test, top10_acc_test] = acc_test
    print("Training accuracy: Top-1:{:.4f}% Top-5:{:.4f}% Top-10:{:.4f}%".format(top1_acc_train * 100., top5_acc_train * 100., top10_acc_train * 100.))
    print("Testing accuracy: Top-1:{:.4f}% Top-5:{:.4f}% Top-10:{:.4f}%".format(top1_acc_test * 100., top5_acc_test * 100., top10_acc_test * 100.))