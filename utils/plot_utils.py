#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_loss_curve(loss_train, loss_val, rounds, args):
        plt.figure(1)
        plt.plot(range(len(loss_train)), loss_train, color='r', label='Training Loss')
        plt.plot(range(len(loss_val)), loss_val, color='b', label='Validation Loss')
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.legend()
        #plt.savefig('./save/loss_round{}_localiter{}_localbs{}.png'.format(rounds, args.local_iter, args.local_bs))
        plt.savefig('./save/LOSS_RoundDuration{}_LocalTrainTime{}_RoundNum{}_LocalIterNum_{}_Localbs{}.png'.format(args.round_duration, args.local_train_time, rounds, args.local_iter, args.local_bs))


def plot_acc_curve(acc_val, rounds, args):
    plt.figure(2)
    plt.plot(range(len(acc_val)), [i[0] for i in acc_val], color='r', label='Top-1')
    plt.plot(range(len(acc_val)), [i[1] for i in acc_val], color='g', label='Top-5')
    plt.plot(range(len(acc_val)), [i[2] for i in acc_val], color='b', label='Top-10')
    plt.xlabel('epoch')
    plt.ylabel('validation_accuracy')
    plt.legend()
    #plt.savefig('./save/acc_round{}_localiter{}_localbs{}.png'.format(rounds, args.local_iter, args.local_bs))
    plt.savefig('./save/ACC_RoundDuration{}_LocalTrainTime{}_RoundNum{}_LocalIterNum_{}_Localbs{}.png'.format(args.round_duration, args.local_train_time, rounds, args.local_iter, args.local_bs))
