#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Bojian
"""
import numpy as np
import argparse
import time
from utils.set import set_random_seed
from utils.eval_metric import eval_metric
from trainer.trainer import train
import pandas as pd
def set_args():
    parser = argparse.ArgumentParser()

    ### Basic parameters
    parser.add_argument('--random_seed', default=2024, type=int, help='random seed')
    parser.add_argument('--backbone_name', default='resnet14', type=str, choices=['resnet14','resnet32', 'resnet18_1D','cnn'], help='the backbone name')
    parser.add_argument('--classifer', default='cos', type=str, choices=['fc', 'cos', 'eu', 'fcwn'], help='the classifier')
    parser.add_argument('--train_parames', default='default', type=str, choices=['default','BN', 'woBN'], help='the trained parameters of model')
    parser.add_argument('--preprocess', default= 'zscore', type=str, choices=['zscore', 'minmax', 'None'], help='the preprocess setting')
    parser.add_argument('--eta_min', default=0.001, type=float, help='the eta_min for CosineAnnealingLR')

    ### Train source parameters
    parser.add_argument('--contrastive_loss', action='store_true', help='the contrastive loss setting')
    parser.add_argument('--LabelSmooth', action='store_true', help='the LabelSmooth setting')
    parser.add_argument('--PCL', action='store_true', help='the PCL setting')
    parser.add_argument('--base_epochs', default=40, type=int, help='the number of epochs in base train')
    parser.add_argument('--base_lr', default=0.1, type=float, help='the learning rate for base train')
    parser.add_argument('--RandMix', action='store_true', help='the RandMix setting')

    ### Incremental parameters
    parser.add_argument('--incremental_mode', default='AFSFFD', type=str, choices=['norm', 'PesudoLabel', 'Tent', 'shot', 'CoTTA', 'gsfda', 'UCSN','CoSDA', 'ours', 'ours_sc', 'ours_simple', 'RaTP', 'EATA', 'AFSFFD'], help='the incremental mode')
    parser.add_argument('--epochs', default=40, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='the learning rate')

    ### Dataset parameters
    parser.add_argument('--batch_size', default=64, type=int, help='the batch size for data loader, for iFlytek, it is 64')
    parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test data loader')
    parser.add_argument('--dataset_name', default='iFlytek', type=str, choices=['SK', 'SK_new', 'iFlytek', 'WT', 'PU_Real', 'PU_Art'], help='the dataset name')
    parser.add_argument('--data_num_class', default=100, type=int, help='the number of data in each class')

    ### No need to set the following parameters 
    parser.add_argument('--data_dimension', type=str, choices=['1D', '2D'], help='the dimension of data')
    parser.add_argument('--data_mode', type=str, choices=['Frequence', 'Time'], help='the mode of data')
    parser.add_argument('--dataroot', default='./data/', type=str, help='the path to load the data')
    parser.add_argument('--nb_cl', type=int, help='the number of classes')
    parser.add_argument('--train_list',type=str, help='the name of the source dir')
    parser.add_argument('--test_list',  type=str, help='the name of the test dir')
    parser.add_argument('--Domain_Seq',  type=int, help='the Domain_Seq setting')
    parser.add_argument('--nb_session', type=int, help='the number of sessions')

    ### Save all
    parser.add_argument('--save_model', action='store_false', help='the save setting')

    ### Ablation setting for ours_simple
    parser.add_argument('--TOPK', action='store_false', help='the ablation setting')
    parser.add_argument('--PCA', action='store_false', help='the ablation setting')
    parser.add_argument('--MI', action='store_false', help='the ablation setting')
    parser.add_argument('--SR', action='store_false', help='the ablation setting')

    ### Ablation setting for ours
    parser.add_argument('--consistency', action='store_false', help='the ablation setting')
    parser.add_argument('--dual', action='store_false', help='the ablation setting')
    parser.add_argument('--mixup', action='store_false', help='the ablation setting')
    parser.add_argument('--rst_min', default=0.0005, type=float, help='Override rst_min value')
    parser.add_argument('--rst_max', default=0.01, type=float, help='Override rst_max value')
    parser.add_argument('--tao_begin', default=0.9, type=float, help='Override tao_begin value')
    parser.add_argument('--tao_end', default=0.99, type=float, help='Override tao_end value')
    ### Get all the arguments
    args = parser.parse_args()

    return args

def set_args_for_different_methods(args):
    ### Set specific arguments for different methods
    if  args.incremental_mode == 'PesudoLabel' or args.incremental_mode == 'Tent' or args.incremental_mode == 'EATA':
        args.train_parames = 'BN'

    if  args.incremental_mode == 'shot':
        args.classifer = 'fcwn'
        args.LabelSmooth = True

    if args.incremental_mode == 'ours_simple':
        args.classifer = 'cos'
        args.PCL = True
        args.LabelSmooth = False
        args.contrastive_loss = False
        # args.TOPK = True
        # args.PCA = True
        # args.MI = True
        # args.SR = True


    if args.incremental_mode == 'gsfda':
        args.classifer = 'fcwn' 

    if args.incremental_mode == 'UCSN':
        args.classifer = 'fc' 

    if args.incremental_mode == 'CoSDA':
        args.classifer = 'fcwn' 

    if args.incremental_mode == 'ours':
        args.classifer = 'cos' 
        args.PCL = False
        args.LabelSmooth = False
        args.contrastive_loss = False

    if args.incremental_mode == 'RaTP':
        args.classifer = 'cos'
        args.PCL = True
        args.RandMix = True

    ### Set specific arguments for different models
    if args.backbone_name == 'cnn' or args.backbone_name == 'resnet18_1D':
        args.data_dimension = '1D'
        args.data_mode = 'Time'

    if args.backbone_name == 'resnet14' or args.backbone_name == 'resnet32':
        args.data_dimension = '2D'
        args.data_mode = 'Frequence'

    ### Set specific arguments for different datasets
    if args.dataset_name == 'SK':
        args.train_list = './SK_all_10classes.mat'
        args.test_list = './SK_all_10classes.mat'
        # args.Domain_Seq = np.array([6,1,8,15,22,17])  # 转速 负载 持续变化
        # args.Domain_Seq = np.array([1,6,8,15,22,17])  # 转速 负载 持续变化
        # args.Domain_Seq = np.array([1,8,15,17])  # 转速 负载 持续变化
        # args.Domain_Seq = np.array([0,1,2,3,4,5])  # 转速 负载 持续变化
        # args.Domain_Seq = np.array([0,1,8,15,22,17])  # 转速 负载 持续变化
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 10

    if args.dataset_name == 'SK_new':
        args.train_list = './SK_new_all_10classes.mat'
        args.test_list = './SK_new_all_10classes.mat'
        # args.Domain_Seq = np.arange(6)
        # args.Domain_Seq = np.array([18,19,20,21,22,23])  
        # args.Domain_Seq = np.array([1,8,15,17])  # 转速 负载 持续变化
        # args.Domain_Seq = np.array([6,1])  # 转速 负载 持续变化
        # args.Domain_Seq = np.array([0,1,8,15,22,17])  # 转速 负载 持续变化
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 10

    if args.dataset_name == 'HUST':
        args.train_list = './HUST_Bearings_10domains_9classes.mat'
        args.test_list = './HUST_Bearings_10domains_9classes.mat'
        # args.Domain_Seq = np.arange(10)
        # args.Domain_Seq = np.array([0,1,2,3])  
        # args.Domain_Seq = np.array([5,6,7,8]) 
        # args.Domain_Seq = np.array([6,1])  # 转速 负载 持续变化
        # args.Domain_Seq = np.array([0,1,8,15,22,17])  # 转速 负载 持续变化
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 9

    if args.dataset_name == 'HUST_gear':
        args.train_list = './HUST_Gearbox_25domains_3classes.mat'
        args.test_list = './HUST_Gearbox_25domains_3classes.mat'

        # args.Domain_Seq = np.arange(5)
        # args.Domain_Seq = np.array([0,6,12,18,24])
        # args.Domain_Seq = np.array([5,6,7,8,9]) 
        # args.Domain_Seq = np.array([6,1])  # 转速 负载 持续变化
        # args.Domain_Seq = np.array([0,1,8,15,22,17])  # 转速 负载 持续变化
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 3

    if args.dataset_name == 'iFlytek':
        args.batch_size = 64
        args.train_list = './iFlytek_all_5classes.mat'
        args.test_list = './iFlytek_all_5classes.mat'
        # args.Domain_Seq = np.array([2,3,4,5,7])  # 声音信号
        # args.Domain_Seq = np.array([1,8,9,10,11,15,14,13,12,3])  # 声音信号
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5

    if args.dataset_name == 'WT':
        args.batch_size = 128
        args.train_list = './WT_all_5classes.mat'
        args.test_list = './WT_all_5classes.mat'
        # args.Domain_Seq = np.array([0,1,2,3,4]) 
        # args.Domain_Seq = np.array([0,1,2,3]) 
        # args.Domain_Seq = np.array([0,1,2,3,4]) 
        # args.Domain_Seq = np.array([5,3,1,2,4]) 
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5

    if args.dataset_name == 'PU_Real':
        args.train_list = './PU_Real_4doamins_5classes.mat'
        args.test_list = './PU_Real_4doamins_5classes.mat'
        # args.base_epochs = 60
        # args.Domain_Seq = np.array([3,2,0,1])  
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5

    if args.dataset_name == 'PU_Real_200':
        args.train_list = './PU_Real_4doamins_5classes_200.mat'
        args.test_list = './PU_Real_4doamins_5classes_200.mat'
        # args.base_epochs = 60
        # args.Domain_Seq = np.array([3,2,0,1])  
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5
        args.data_num_class = 200

    if args.dataset_name == 'PU_Art':
        args.train_list = './PU_Art_4doamins_8classes.mat'
        args.test_list = './PU_Art_4doamins_8classes.mat'
        # args.Domain_Seq = np.array([1,3,2,0])  
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 8

    if args.dataset_name == 'Robot':
        args.train_list = './Robot_4classes_4domains.mat'
        args.test_list = './Robot_4classes_4domains.mat'
        # args.Domain_Seq = np.array([0,1,2,3])  
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 4


    return args

# Domain_
def set_domain_seq(dataset_name):
    if dataset_name == 'SK':
        Seq =  [np.array([6,1,8,15,22,17]),
            np.array([1,6,8,15,22,17]),
            np.array([6,8,1,15,22,17]),
            np.array([6,1,15,8,22,17]),
            np.array([6,1,8,22,15,17]),
            np.array([6,1,8,15,17,22])
            ]
    elif dataset_name == 'iFlytek':

        Seq =  [np.array([2,3,4,5,7]),
                np.array([3,2,4,5,7]),
                np.array([2,4,3,5,7]),
                np.array([2,3,5,4,7]),
                np.array([2,3,4,7,5])]
        
    elif dataset_name == 'WT':
        Seq =  [np.array([0,1,2,3,4]),
                np.array([1,0,2,3,4]),
                np.array([0,2,1,3,4]),
                np.array([0,1,3,2,4]),
                np.array([0,1,2,4,3])]
    return Seq
#     np.array([0,3,2,1]),
#     np.array([1,0,2,3]),
#     np.array([1,0,3,2]),
#     np.array([1,2,0,3]),
#     np.array([1,2,3,0]),
#     np.array([1,3,0,2]),
#     np.array([1,3,2,0]),
#     np.array([2,0,1,3]),
#     np.array([2,0,3,1]),
#     np.array([2,1,0,3]),
#     np.array([2,1,3,0]),
#     np.array([2,3,0,1]),
#     np.array([2,3,1,0]),
#     np.array([3,0,1,2]),
#     np.array([3,0,2,1]),
#     np.array([3,1,0,2]),
#     np.array([3,1,2,0]),
#     np.array([3,2,0,1]),
#     np.array([3,2,1,0]),
    # 
# # Seed
Seed = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
# Seed = [2024]
# Mode
# Mode = ['ours']
# Mode = []
# args = set_args()
# set_random_seed(args.random_seed)
Dataset_name = ['iFlytek']
# Dataset_name = ['SK','iFlytek']
for dataset_name in Dataset_name:
    # if dataset_name == 'WT':
    #     Mode = ['norm', 'PesudoLabel', 'Tent', 'shot', 'CoTTA', 'gsfda', 'UCSN','CoSDA', 'ours', 'ours_simple', 'EATA', 'AFSFFD']
    #     Mode = ['ours']
    # else:
    # Mode = ['norm', 'PesudoLabel', 'Tent', 'shot', 'CoTTA', 'gsfda', 'UCSN','CoSDA', 'ours_simple', 'EATA', 'AFSFFD']
    Mode = ["ours_new"]
    for mode in Mode:
        Results_list = []
        Seq = set_domain_seq(dataset_name)
        for seq in Seq:

            for seed in Seed:

                args = set_args()
                args.dataset_name = dataset_name
                args.incremental_mode = mode
                args.Domain_Seq = seq
                args.random_seed = seed
                args = set_args_for_different_methods(args)
                set_random_seed(args.random_seed)
                print('=============================================================================================================')
                print(args)
                print('=============================================================================================================')
                print( args.dataset_name)
                print('=============================================================================================================')
                print('Start Training')
                print('=============================================================================================================')

                Correct = train(args)
                AP, AF, AMF, AG, AA, BWT, ACC  = eval_metric(args, Correct)
                # Create a dictionary to store the results
                results_dict = {
                    'Seq': seq,
                    'Seed': seed,
                    'AP': AP,
                    'AF': AF,
                    'AMF': AMF,
                    'AG': AG,
                    'AA': AA,
                    'BWT': BWT,
                    'ACC': ACC
                }
                Results_list.append(results_dict)
        results_df = pd.DataFrame(Results_list)
        
        # Calculate mean and std deviation for numeric columns
        numeric_cols = ['AP', 'AF', 'AMF', 'AG', 'AA', 'BWT', 'ACC']
        mean_row = {}
        std_row = {}
        
        for col in numeric_cols:
            mean_row[col] = results_df[col].mean()
            std_row[col] = results_df[col].std()
        
        # Set non-numeric columns to appropriate values
        mean_row['Seq'] = 'Mean'
        mean_row['Seed'] = ''
        std_row['Seq'] = 'Std'
        std_row['Seed'] = ''
        
        # Append mean and std rows to the dataframe
        results_df = pd.concat([results_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
        results_pth = './log/' + args.dataset_name + '/' + args.incremental_mode + '/' + args.preprocess + '/results_parallel_' + args.incremental_mode + '.csv'
        results_df.to_csv(results_pth, index=False)

