#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Bojian
Grid search sensitivity experiment for hyperparameters: rst_min, rst_max, tao_begin, tao_end
"""
import numpy as np
import argparse
import time
import os
import pandas as pd
import itertools
from datetime import datetime
from utils.set import set_random_seed
from utils.eval_metric import eval_metric
from trainer.trainer import train

# ============ Hyperparameter Grid Definition ============
RST_MIN_VALUES = [0.0005, 0.001, 0.005, 0.01]
RST_MAX_VALUES = [0.005, 0.01, 0.05, 0.1]
TAO_BEGIN_VALUES = [0.9, 0.95, 0.97, 0.99]
TAO_END_VALUES = [0.95, 0.97, 0.99, 0.999]

# Default values (from ours_new.py)
DEFAULT_RST_MIN = 0.001
DEFAULT_RST_MAX = 0.01
DEFAULT_TAO_BEGIN = 0.95
DEFAULT_TAO_END = 0.99

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
parser.add_argument('--incremental_mode', default='ours_new', type=str, choices=['norm', 'PesudoLabel', 'Tent', 'shot', 'CoTTA', 'gsfda', 'UCSN' 'CoSDA', 'ours', 'ours_simple', 'RaTP', 'EATA', 'AFSFFD', 'ours_new'], help='the incremental mode')
parser.add_argument('--epochs', default=40, type=int, help='the number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='the learning rate')

### Dataset parameters
parser.add_argument('--batch_size', default=64, type=int, help='the batch size for data loader, for iFlytek, it is ·64')
parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test data loader')
parser.add_argument('--dataset_name', default='WT', type=str, choices=['SK', 'SK_new', 'iFlytek', 'WT', 'PU_Real', 'PU_Art'], help='the dataset name')
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

### ours_new ablation flags
parser.add_argument('--CKCR', action='store_true', help='Enable CKCR')
parser.add_argument('--select_soft_knowledge', action='store_true', help='Enable soft knowledge selection')
parser.add_argument('--mixup', action='store_true', help='Enable mixup')
parser.add_argument('--EPHS', action='store_true', help='Enable EPHS')
parser.add_argument('--FISR', action='store_true', help='Enable Fisher-weighted SR')
parser.add_argument('--SR', action='store_true', help='Enable basic SR')
parser.add_argument('--MI', action='store_true', help='Enable MI loss')
parser.add_argument('--TOPK', action='store_false', help='the ablation setting')
parser.add_argument('--PCA', action='store_false', help='the ablation setting')

### Grid search parameters
parser.add_argument('--rst_min', default=None, type=float, help='Override rst_min value')
parser.add_argument('--rst_max', default=None, type=float, help='Override rst_max value')
parser.add_argument('--tao_begin', default=None, type=float, help='Override tao_begin value')
parser.add_argument('--tao_end', default=None, type=float, help='Override tao_end value')

### Sensitivity experiment mode
parser.add_argument('--sensitivity_mode', default='full', type=str, 
                    choices=['full', 'rst_min', 'rst_max', 'tao_begin', 'tao_end'],
                    help='Sensitivity mode: full=all combinations, or specify single parameter to vary')
parser.add_argument('--quick_test', action='store_true', help='Quick test mode with minimal grid')
parser.add_argument('--output_dir', default='./log/sensitivity/', type=str, help='Output directory for results')

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

if args.incremental_mode == 'ours_new':
    args.classifer = 'cos'
    args.PCL = False
    args.LabelSmooth = False
    args.contrastive_loss = False
    # Set default ours_new ablation flags for full model
    args.CKCR = True
    args.select_soft_knowledge = True
    args.mixup = True
    args.MI = True
    args.EPHS = True
    args.FISR = True
    args.SR = False

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
    args.Domain_Seq = np.array([6,1,8,15,22,17])
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 10

if args.dataset_name == 'SK_new':
    args.train_list = './SK_new_all_10classes.mat'
    args.test_list = './SK_new_all_10classes.mat'
    args.Domain_Seq = np.array([6,1,8,15,22,17]) 
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 10

if args.dataset_name == 'HUST':
    args.train_list = './HUST_Bearings_10domains_9classes.mat'
    args.test_list = './HUST_Bearings_10domains_9classes.mat'
    args.Domain_Seq = np.array([5,6,7,8]) 
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 9

if args.dataset_name == 'HUST_gear':
    args.train_list = './HUST_Gearbox_25domains_3classes.mat'
    args.test_list = './HUST_Gearbox_25domains_3classes.mat'
    args.Domain_Seq = np.array([0,6,12,18,24])
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 3

if args.dataset_name == 'iFlytek':
    args.batch_size = 64
    args.train_list = './iFlytek_all_5classes.mat'
    args.test_list = './iFlytek_all_5classes.mat'
    args.Domain_Seq = np.array([2,3,4,5,7])
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 5

if args.dataset_name == 'WT':
    args.batch_size = 128
    args.train_list = './WT_all_5classes.mat'
    args.test_list = './WT_all_5classes.mat'
    args.Domain_Seq = np.array([0,1,2,3,4]) 
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 5

if args.dataset_name == 'PU_Real':
    args.train_list = './PU_Real_4doamins_5classes.mat'
    args.test_list = './PU_Real_4doamins_5classes.mat'
    args.Domain_Seq = np.array([3,2,0,1])  
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 5

if args.dataset_name == 'PU_Real_200':
    args.train_list = './PU_Real_4doamins_5classes_200.mat'
    args.test_list = './PU_Real_4doamins_5classes_200.mat'
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


def generate_param_combinations(args):
    """Generate hyperparameter combinations based on sensitivity mode."""
    
    if args.quick_test:
        # Quick test: only 2 values for each to reduce combinations
        rst_min_vals = [0.001, 0.01]
        rst_max_vals = [0.01, 0.05]
        tao_begin_vals = [0.95, 0.99]
        tao_end_vals = [0.99, 0.999]
    else:
        rst_min_vals = RST_MIN_VALUES
        rst_max_vals = RST_MAX_VALUES
        tao_begin_vals = TAO_BEGIN_VALUES
        tao_end_vals = TAO_END_VALUES
    
    if args.sensitivity_mode == 'full':
        # Full grid search
        combinations = list(itertools.product(rst_min_vals, rst_max_vals, tao_begin_vals, tao_end_vals))
        # Filter invalid combinations: rst_min < rst_max, tao_begin < tao_end
        combinations = [(r_min, r_max, t_begin, t_end) 
                        for r_min, r_max, t_begin, t_end in combinations 
                        if r_min < r_max and t_begin < t_end]
    elif args.sensitivity_mode == 'rst_min':
        combinations = [(r_min, DEFAULT_RST_MAX, DEFAULT_TAO_BEGIN, DEFAULT_TAO_END) 
                        for r_min in rst_min_vals if r_min < DEFAULT_RST_MAX]
    elif args.sensitivity_mode == 'rst_max':
        combinations = [(DEFAULT_RST_MIN, r_max, DEFAULT_TAO_BEGIN, DEFAULT_TAO_END) 
                        for r_max in rst_max_vals if DEFAULT_RST_MIN < r_max]
    elif args.sensitivity_mode == 'tao_begin':
        combinations = [(DEFAULT_RST_MIN, DEFAULT_RST_MAX, t_begin, DEFAULT_TAO_END) 
                        for t_begin in tao_begin_vals if t_begin < DEFAULT_TAO_END]
    elif args.sensitivity_mode == 'tao_end':
        combinations = [(DEFAULT_RST_MIN, DEFAULT_RST_MAX, DEFAULT_TAO_BEGIN, t_end) 
                        for t_end in tao_end_vals if DEFAULT_TAO_BEGIN < t_end]
    else:
        raise ValueError(f"Unknown sensitivity mode: {args.sensitivity_mode}")
    
    return combinations


def run_experiment(args, rst_min, rst_max, tao_begin, tao_end):
    """Run a single experiment with given hyperparameters."""
    
    # Set hyperparameters
    args.rst_min = rst_min
    args.rst_max = rst_max
    args.tao_begin = tao_begin
    args.tao_end = tao_end
    
    print(f"\n{'='*80}")
    print(f"Running experiment: rst_min={rst_min}, rst_max={rst_max}, tao_begin={tao_begin}, tao_end={tao_end}")
    print(f"{'='*80}")
    
    # Run training
    Correct = train(args)
    
    # Compute metrics
    AP, AF, AMF, AG, AA, BWT, ACC = eval_metric(args, Correct)
    
    return {
        'rst_min': rst_min,
        'rst_max': rst_max,
        'tao_begin': tao_begin,
        'tao_end': tao_end,
        'AP': AP,
        'AF': AF,
        'AMF': AMF,
        'AG': AG,
        'AA': AA,
        'BWT': BWT,
        'ACC': ACC
    }


def main():
    """Main function for grid search sensitivity experiment."""
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.dataset_name, args.sensitivity_mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate parameter combinations
    combinations = generate_param_combinations(args)
    print(f"\nTotal combinations to test: {len(combinations)}")
    print(f"Sensitivity mode: {args.sensitivity_mode}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output directory: {output_dir}")
    
    # Store results
    results = []
    
    # Run experiments
    time_start = time.time()
    for idx, (rst_min, rst_max, tao_begin, tao_end) in enumerate(combinations):
        print(f"\n>>> Experiment {idx+1}/{len(combinations)}")
        
        try:
            result = run_experiment(args, rst_min, rst_max, tao_begin, tao_end)
            results.append(result)
            
            # Save intermediate results after each experiment
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = os.path.join(output_dir, f'sensitivity_results_intermediate_{timestamp}.csv')
            df.to_csv(intermediate_file, index=False)
            print(f"Intermediate results saved to: {intermediate_file}")
            
        except Exception as e:
            print(f"Error in experiment: {e}")
            results.append({
                'rst_min': rst_min,
                'rst_max': rst_max,
                'tao_begin': tao_begin,
                'tao_end': tao_end,
                'AP': None,
                'AF': None,
                'AMF': None,
                'AG': None,
                'AA': None,
                'BWT': None,
                'ACC': None,
                'error': str(e)
            })
    
    time_end = time.time()
    
    # Save final results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = os.path.join(output_dir, f'sensitivity_results_final_{timestamp}.csv')
    df.to_csv(final_file, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SENSITIVITY EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {(time_end - time_start)/60:.2f} minutes")
    print(f"Results saved to: {final_file}")
    
    # Print best result
    if results:
        valid_results = [r for r in results if r.get('ACC') is not None]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['ACC'])
            print(f"\nBest configuration (by ACC):")
            print(f"  rst_min={best_result['rst_min']}, rst_max={best_result['rst_max']}")
            print(f"  tao_begin={best_result['tao_begin']}, tao_end={best_result['tao_end']}")
            print(f"  ACC={best_result['ACC']:.2f}%")
    
    print(f"\nAll results:")
    print(df.to_string())
    
    return df


if __name__ == '__main__':
    main()
