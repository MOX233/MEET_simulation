#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np

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
    

def Raymobtime_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from Raymobtime dataset
    :param dataset:
    :param num_users:
    :return:
    """

    pass

    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
    """
