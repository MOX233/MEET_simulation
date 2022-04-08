#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import os
import pickle as pkl

from cv2 import log

def save_training_log(args, loss_train, loss_val, acc_val):
    log_dict = {}
    log_dict['args'] = args
    log_dict['loss_train'] = loss_train
    log_dict['loss_val'] = loss_val
    log_dict['acc_val'] = acc_val
    log_dict.update(vars(args))
    log_dict['device'] = str(log_dict['device'])
    savePath = "./save"
    if args.log_save_path != "default":
            savePath = args.log_save_path
    os.makedirs(savePath, exist_ok=True)
    savePath = os.path.join(savePath,'RoundDuration{}_LocalTrainDelay_mu{}_beta{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}_SplitDict{}.pkl'.format(args.round_duration, args.mu_local_train, args.beta_local_train, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, args.non_iid, args.split_dict))
    with open(savePath,'wb') as f:
        pkl.dump(log_dict, f)

def load_training_log(savePath):
    with open(savePath,'rb') as f:
        log_dict = pkl.load(f)
    return log_dict