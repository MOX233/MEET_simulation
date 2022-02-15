#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from functools import partial
import numpy as np
import csv

def get_gps_data(gps_data_path, data_range=[0,None], x_split_num=1, y_split_num=1):
    gps_data = []
    with open(gps_data_path,'r') as f:
        csv_reader = csv.reader(f)
        for x in csv_reader:
            if x[0]=='V':   # we just preserve the data with valid channel
                gps_data.append(x)
    gps_data = gps_data[data_range[0]:data_range[1]]
    LOS_split_dict = dict(LOS=[], NLOS=[])
    for idx, x in enumerate(gps_data):
        if x[9]=='LOS=0':
            LOS_split_dict['NLOS'].append(idx)
        if x[9]=='LOS=1':
            LOS_split_dict['LOS'].append(idx)
    xmin, xmax = 10000, 0
    ymin, ymax = 10000, 0
    for x in gps_data:
        xmax = max(xmax,float(x[5]))
        xmin = min(xmin,float(x[5]))
        ymax = max(ymax,float(x[6]))
        ymin = min(ymin,float(x[6]))
    #(xmin,xmax,ymin,ymax) == (745.2150970651, 765.94906193325, 431.1722565652, 676.9144333766001)
    coord_split_dict = {}
    # split the dataset by coordinate
    x_split_points = np.linspace(xmin,xmax,x_split_num+1)
    y_split_points = np.linspace(ymin,ymax,y_split_num+1)
    for x_idx in range(x_split_num):
        for y_idx in range(y_split_num):
            coord_split_dict[x_idx*y_split_num + y_idx] = []
            for idx, x in enumerate(gps_data):
                if float(x[5])>=x_split_points[x_idx] and float(x[5])<=x_split_points[x_idx+1] and float(x[6])>=y_split_points[y_idx] and float(x[6])<=y_split_points[y_idx+1] :
                    coord_split_dict[x_idx*y_split_num + y_idx].append(idx)
    return gps_data, LOS_split_dict, coord_split_dict

def Raymobtime_iid(dataset, num_items, num_users=1):
    """
    Sample I.I.D. client data from Raymobtime dataset
    :param dataset:
    :param num_items: specify every user's local dataset size. type: int or list
    :param num_users:
    :return: dict of image index
    """
    if type(num_items) == int:
        assert num_items>0
        num_items = min([num_items, len(dataset)])
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        return dict_users
    elif type(num_items) == list:
        assert len(num_items) == num_users
        assert max(num_items) <= len(dataset)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
        return dict_users
    else:
        exit('The type of num_items is wrong!')
    

def Raymobtime_noniid(dataset, split_dict, num_items, num_users):
    """
    Sample non-I.I.D client data from Raymobtime dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if type(num_items) == int:
        assert num_items>0
        dict_users = {}
        for i in range(num_users):
            key_table = list(split_dict.keys())
            partial_idxs = split_dict[key_table[np.random.randint(len(key_table))]]
            dict_users[i] = set(np.random.choice(partial_idxs, num_items, replace=True))
        return dict_users
    elif type(num_items) == list:
        assert len(num_items) == num_users
        dict_users = {}
        for i in range(num_users):
            key_table = list(split_dict.keys())
            partial_idxs = split_dict[key_table[np.random.randint(len(key_table))]]
            dict_users[i] = set(np.random.choice(partial_idxs, num_items[i], replace=True))
        return dict_users
    else:
        exit('The type of num_items is wrong!')