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

parser = argparse.ArgumentParser()

### Basic parameters
parser.add_argument('--random_seed', default=2025, type=int, help='random seed')
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
parser.add_argument('--incremental_mode', default='ours_new', type=str, choices=['norm', 'PesudoLabel', 'Tent', 'shot', 'CoTTA', 'gsfda', 'UCSN' 'CoSDA', 'ours', 'ours_simple', 'RaTP', 'EATA', 'AFSFFD'], help='the incremental mode')
parser.add_argument('--epochs', default=40, type=int, help='the number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='the learning rate')

### Dataset parameters
parser.add_argument('--batch_size', default=64, type=int, help='the batch size for data loader, for iFlytek, it is ·64')
parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test data loader')
parser.add_argument('--dataset_name', default='SK', type=str, choices=['SK', 'SK_new', 'iFlytek', 'WT', 'PU_Real', 'PU_Art'], help='the dataset name')
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
parser.add_argument('--save_model', action='store_true', help='the save setting')

### Ablation setting for ours_simple
parser.add_argument('--CKCR', action='store_false', help='Enable CKCR')
parser.add_argument('--select_soft_knowledge', action='store_false', help='Enable soft knowledge selection')
parser.add_argument('--mixup', action='store_false', help='Enable mixup')
parser.add_argument('--EPHS', action='store_false', help='Enable EPHS')
parser.add_argument('--FISR', action='store_false', help='Enable Fisher-weighted SR')
parser.add_argument('--SR', action='store_false', help='Enable basic SR')
parser.add_argument('--MI', action='store_false', help='Enable MI loss')
parser.add_argument('--EMA', action='store_false', help='Enable EMA for Mean-Teacher')
parser.add_argument('--Reset_student', action='store_false', help='Reset student from teacher each epoch')



### Hyperparameters for ours_new
parser.add_argument('--rst_min', default=None, type=float, help='Override rst_min value')
parser.add_argument('--rst_max', default=None, type=float, help='Override rst_max value')
parser.add_argument('--tao_begin', default=None, type=float, help='Override tao_begin value')
parser.add_argument('--tao_end', default=None, type=float, help='Override tao_end value')




### Get all the arguments
args = parser.parse_args()

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
if args.incremental_mode == 'ours_sc':
    args.classifer = 'cos' 
    args.PCL = False
    args.LabelSmooth = False
    args.contrastive_loss = False

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
    args.Domain_Seq = np.array([6,1,8,15,22,17])  # 转速 负载 持续变化
    # args.Domain_Seq = np.array([1,6,8,15,22,17])  # 转速 负载 持续变化
    # args.Domain_Seq = np.array([1,8,15,17])  # 转速 负载 持续变化
    # args.Domain_Seq = np.array([0,1,2,3,4,5])  # 转速 负载 持续变化
    # args.Domain_Seq = np.array([0,1,8,15,22,17])  # 转速 负载 持续变化
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 10
    # Grid-search optimized hyperparameters for SK
    args.tao_begin = 0.95
    args.tao_end = 0.99
    args.rst_min = 0.005
    args.rst_max = 0.05

if args.dataset_name == 'SK_new':
    args.train_list = './SK_new_all_10classes.mat'
    args.test_list = './SK_new_all_10classes.mat'
    # args.Domain_Seq = np.arange(6)
    args.Domain_Seq = np.array([6,1,8,15,22,17]) 
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
    args.Domain_Seq = np.array([5,6,7,8]) 
    # args.Domain_Seq = np.array([6,1])  # 转速 负载 持续变化
    # args.Domain_Seq = np.array([0,1,8,15,22,17])  # 转速 负载 持续变化
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 9

if args.dataset_name == 'HUST_gear':
    args.train_list = './HUST_Gearbox_25domains_3classes.mat'
    args.test_list = './HUST_Gearbox_25domains_3classes.mat'

    # args.Domain_Seq = np.arange(5)
    args.Domain_Seq = np.array([0,6,12,18,24])
    # args.Domain_Seq = np.array([5,6,7,8,9]) 
    # args.Domain_Seq = np.array([6,1])  # 转速 负载 持续变化
    # args.Domain_Seq = np.array([0,1,8,15,22,17])  # 转速 负载 持续变化
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 3

if args.dataset_name == 'iFlytek':
    args.batch_size = 64
    args.train_list = './iFlytek_all_5classes.mat'
    args.test_list = './iFlytek_all_5classes.mat'
    args.Domain_Seq = np.array([2,3,4,5,7])  # 声音信号
    # args.Domain_Seq = np.array([1,8,9,10,11,15,14,13,12,3])  # 声音信号
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 5
    # Grid-search optimized hyperparameters for iFlytek
    args.tao_begin = 0.95
    args.tao_end = 0.97
    args.rst_min = 0.0005
    args.rst_max = 0.1

if args.dataset_name == 'WT':
    args.batch_size = 128
    args.train_list = './WT_all_5classes.mat'
    args.test_list = './WT_all_5classes.mat'
    # args.Domain_Seq = np.array([0,1,2,3,4]) 
    # args.Domain_Seq = np.array([0,1,2,3]) 
    args.Domain_Seq = np.array([0,1,2,3,4]) 
    # args.Domain_Seq = np.array([5,3,1,2,4]) 
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 5
    # Grid-search optimized hyperparameters for WT
    args.tao_begin = 0.95
    args.tao_end = 0.97
    args.rst_min = 0.01
    args.rst_max = 0.05

if args.dataset_name == 'PU_Real':
    args.train_list = './PU_Real_4doamins_5classes.mat'
    args.test_list = './PU_Real_4doamins_5classes.mat'
    # args.base_epochs = 60
    args.Domain_Seq = np.array([3,2,0,1])  
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 5

if args.dataset_name == 'PU_Real_200':
    args.train_list = './PU_Real_4doamins_5classes_200.mat'
    args.test_list = './PU_Real_4doamins_5classes_200.mat'
    # args.base_epochs = 60
    args.Domain_Seq = np.array([3,2,0,1])  
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 5
    args.data_num_class = 200

if args.dataset_name == 'PU_Art':
    args.train_list = './PU_Art_4doamins_8classes.mat'
    args.test_list = './PU_Art_4doamins_8classes.mat'
    args.Domain_Seq = np.array([2,0])  
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 8

if args.dataset_name == 'Robot':
    args.train_list = './Robot_4classes_4domains.mat'
    args.test_list = './Robot_4classes_4domains.mat'
    args.Domain_Seq = np.array([0,1,2,3])  
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 4


### Set the random seed
set_random_seed(args.random_seed)

print('=============================================================================================================')
print(args)
print('=============================================================================================================')
print( args.dataset_name)
print('=============================================================================================================')
print('Start Training')
print('=============================================================================================================')
# time_start=time.time()
Correct = train(args)
# time_end=time.time()
eval_metric(args, Correct)
print(args.incremental_mode)  

