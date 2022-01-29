#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import os
import matplotlib # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt # noqa

def plot_loss_curve(loss_train, loss_val, rounds, args):
        plt.figure(1)
        plt.plot(range(len(loss_train)), loss_train, color='r', label='Training Loss')
        plt.plot(range(len(loss_val)), loss_val, color='b', label='Validation Loss')
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.legend()
        #plt.savefig('./save/loss_round{}_localiter{}_localbs{}.png'.format(rounds, args.local_iter, args.local_bs))
        plt.savefig('./save/LOSS_RoundDuration{}_LocalTrainTime{}_RoundNum{}_LocalIterNum_{}_Localbs{}.png'.format(args.round_duration, args.local_train_time, rounds, args.local_iter, args.local_bs))
        plt.close()


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
        plt.close()

def plot_loss_acc_curve(loss_train, loss_val, acc_val, rounds, args):
        plt.figure(figsize=(5,15),dpi=100)
        # loss
        plt.subplot(3,1,1)
        plt.plot(range(len(loss_train)), loss_train, color='r', label='Training Loss')
        plt.plot(range(len(loss_val)), loss_val, color='b', label='Validation Loss')
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.legend()
        # accuracy
        plt.subplot(3,1,2)
        plt.plot(range(len(acc_val)), [i[0] for i in acc_val], color='r', label='Top-1')
        plt.plot(range(len(acc_val)), [i[1] for i in acc_val], color='g', label='Top-5')
        plt.plot(range(len(acc_val)), [i[2] for i in acc_val], color='b', label='Top-10')
        plt.xlabel('epoch')
        plt.ylabel('validation_accuracy')
        plt.legend()
        # params description
        plt.subplot(3,1,3)
        plt.axis([0,10,0,10])
        plt.axis('off')
        fontsize=12
        plt.text(0,9,'Round Num: {}'.format(rounds),fontsize=fontsize)
        plt.text(0,8,'Round Duration: {}'.format(args.round_duration),fontsize=fontsize)
        plt.text(0,7,'Local Train Delay: {}'.format(args.local_train_time),fontsize=fontsize)
        plt.text(0,6,'Local Iter Num: {}'.format(args.local_iter),fontsize=fontsize)
        plt.text(0,5,'Local Batch Size: {}'.format(args.local_bs),fontsize=fontsize)
        plt.text(0,4,'Learning Rate: {}'.format(args.lr),fontsize=fontsize)
        plt.text(6,9,'Lambda: {}'.format(args.Lambda),fontsize=fontsize)
        plt.text(6,8,'maxSpeed: {}'.format(args.maxSpeed),fontsize=fontsize)
        plt.text(6,7,'beta_download: {}'.format(args.beta_download),fontsize=fontsize)
        plt.text(6,6,'mu_download: {}'.format(args.mu_download),fontsize=fontsize)
        plt.text(6,5,'beta_upload: {}'.format(args.beta_upload),fontsize=fontsize)
        plt.text(6,4,'mu_upload: {}'.format(args.mu_upload),fontsize=fontsize)
        
        savePath = "./save"
        if args.plot_save_path != "default":
                savePath = args.plot_save_path
        savePath = os.path.join(savePath,'RoundNum{}_RoundDuration{}_LocalTrainDelay{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}.png'.format(rounds, args.round_duration, args.local_train_time, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed))
        plt.savefig(savePath)
        plt.close()