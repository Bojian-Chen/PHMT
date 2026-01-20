#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Bojian
Ablation study for ours_new method.
Ablatable components:
  1. Mixup - 数据混合增强
  2. MI - 互信息损失
  3. EMA - 指数移动平均更新
  4. SR - Fisher加权随机恢复
  5. Adaptive - 自适应置信度/参数调节
"""
import numpy as np
import argparse
import time
import os
import pandas as pd
from datetime import datetime
from utils.set import set_random_seed
from utils.eval_metric import eval_metric
from trainer.trainer import train

# ============ Ablation Configurations ============
# Each config: (name, description, args_dict)
ABLATION_CONFIGS = [
    # Full model (baseline)
    ('Full', 'Full model with all components', {
        'ablation_mixup': True,
        'ablation_MI': True,
        'ablation_EMA': True,
        'ablation_SR': True,
        'ablation_adaptive': True,
    }),
    
    # Remove individual components
    ('wo_Mixup', 'Without Mixup augmentation', {
        'ablation_mixup': False,
        'ablation_MI': True,
        'ablation_EMA': True,
        'ablation_SR': True,
        'ablation_adaptive': True,
    }),
    
    ('wo_MI', 'Without Mutual Information loss', {
        'ablation_mixup': True,
        'ablation_MI': False,
        'ablation_EMA': True,
        'ablation_SR': True,
        'ablation_adaptive': True,
    }),
    
    ('wo_EMA', 'Without EMA update', {
        'ablation_mixup': True,
        'ablation_MI': True,
        'ablation_EMA': False,
        'ablation_SR': True,
        'ablation_adaptive': True,
    }),
    
    ('wo_SR', 'Without Stochastic Recovery', {
        'ablation_mixup': True,
        'ablation_MI': True,
        'ablation_EMA': True,
        'ablation_SR': False,
        'ablation_adaptive': True,
    }),
    
    ('wo_Adaptive', 'Without Adaptive parameter adjustment', {
        'ablation_mixup': True,
        'ablation_MI': True,
        'ablation_EMA': True,
        'ablation_SR': True,
        'ablation_adaptive': False,
    }),
    
    # Progressive ablation (cumulative removal)
    ('only_KD', 'Only Knowledge Distillation (no Mixup, MI, EMA, SR, Adaptive)', {
        'ablation_mixup': False,
        'ablation_MI': False,
        'ablation_EMA': False,
        'ablation_SR': False,
        'ablation_adaptive': False,
    }),
    
    ('KD_Mixup', 'KD + Mixup only', {
        'ablation_mixup': True,
        'ablation_MI': False,
        'ablation_EMA': False,
        'ablation_SR': False,
        'ablation_adaptive': False,
    }),
    
    ('KD_Mixup_MI', 'KD + Mixup + MI', {
        'ablation_mixup': True,
        'ablation_MI': True,
        'ablation_EMA': False,
        'ablation_SR': False,
        'ablation_adaptive': False,
    }),
    
    ('KD_Mixup_MI_EMA', 'KD + Mixup + MI + EMA', {
        'ablation_mixup': True,
        'ablation_MI': True,
        'ablation_EMA': True,
        'ablation_SR': False,
        'ablation_adaptive': False,
    }),
    
    ('KD_Mixup_MI_EMA_SR', 'KD + Mixup + MI + EMA + SR (no Adaptive)', {
        'ablation_mixup': True,
        'ablation_MI': True,
        'ablation_EMA': True,
        'ablation_SR': True,
        'ablation_adaptive': False,
    }),
]


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
parser.add_argument('--incremental_mode', default='ours_new', type=str, help='the incremental mode')
parser.add_argument('--epochs', default=40, type=int, help='the number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='the learning rate')

### Dataset parameters
parser.add_argument('--batch_size', default=64, type=int, help='the batch size for data loader')
parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test data loader')
parser.add_argument('--dataset_name', default='SK', type=str, choices=['SK', 'SK_new', 'iFlytek', 'WT', 'PU_Real', 'PU_Art', 'HUST'], help='the dataset name')
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

### Ablation settings (controlled by ABLATION_CONFIGS)
parser.add_argument('--mixup', action='store_false', help='the mixup setting')
parser.add_argument('--MI', action='store_false', help='the MI loss setting')
parser.add_argument('--TOPK', action='store_false', help='the ablation setting')
parser.add_argument('--PCA', action='store_false', help='the ablation setting')
parser.add_argument('--SR', action='store_false', help='the ablation setting')

### Ablation experiment settings
parser.add_argument('--ablation_config', default=None, type=str, help='Specific ablation config to run (e.g., "wo_Mixup")')
parser.add_argument('--output_dir', default='./log/ablation/', type=str, help='Output directory for results')
parser.add_argument('--run_all', action='store_true', help='Run all ablation configurations')

### Hyperparameters (use defaults from sensitivity experiments)
parser.add_argument('--rst_min', default=0.001, type=float, help='rst_min for SR')
parser.add_argument('--rst_max', default=0.01, type=float, help='rst_max for SR')
parser.add_argument('--tao_begin', default=0.95, type=float, help='tao_begin for EMA')
parser.add_argument('--tao_end', default=0.99, type=float, help='tao_end for EMA')

### Ablation control flags
parser.add_argument('--ablation_mixup', default=True, type=lambda x: x.lower() == 'true', help='Enable Mixup')
parser.add_argument('--ablation_MI', default=True, type=lambda x: x.lower() == 'true', help='Enable MI loss')
parser.add_argument('--ablation_EMA', default=True, type=lambda x: x.lower() == 'true', help='Enable EMA')
parser.add_argument('--ablation_SR', default=True, type=lambda x: x.lower() == 'true', help='Enable SR')
parser.add_argument('--ablation_adaptive', default=True, type=lambda x: x.lower() == 'true', help='Enable Adaptive')


### Get all the arguments
args = parser.parse_args()

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

if args.dataset_name == 'PU_Art':
    args.train_list = './PU_Art_4doamins_8classes.mat'
    args.test_list = './PU_Art_4doamins_8classes.mat'
    args.Domain_Seq = np.array([2,0])  
    args.nb_session = len(args.Domain_Seq)
    args.nb_cl = 8


def apply_ablation_config(args, config_dict):
    """Apply ablation configuration to args."""
    # Map ablation flags to actual model parameters
    args.mixup = config_dict.get('ablation_mixup', True)
    args.MI = config_dict.get('ablation_MI', True)
    args.ablation_EMA = config_dict.get('ablation_EMA', True)
    args.ablation_SR = config_dict.get('ablation_SR', True)
    args.ablation_adaptive = config_dict.get('ablation_adaptive', True)
    
    # If EMA is disabled, set tao to 0 (no momentum)
    if not args.ablation_EMA:
        args.tao_begin = 0.0
        args.tao_end = 0.0
    
    # If SR is disabled, set rst to 0 (no recovery)
    if not args.ablation_SR:
        args.rst_min = 0.0
        args.rst_max = 0.0
    
    # If Adaptive is disabled, use fixed values
    if not args.ablation_adaptive:
        args.fixed_conf = True
    else:
        args.fixed_conf = False
    
    return args


def run_single_ablation(args, config_name, config_desc, config_dict):
    """Run a single ablation experiment."""
    print(f"\n{'='*80}")
    print(f"Ablation: {config_name}")
    print(f"Description: {config_desc}")
    print(f"Config: {config_dict}")
    print(f"{'='*80}")
    
    # Apply configuration
    args = apply_ablation_config(args, config_dict)
    
    # Run training
    Correct = train(args)
    
    # Compute metrics
    AP, AF, AMF, AG, AA, BWT, ACC = eval_metric(args, Correct)
    
    return {
        'config_name': config_name,
        'config_desc': config_desc,
        'mixup': config_dict.get('ablation_mixup', True),
        'MI': config_dict.get('ablation_MI', True),
        'EMA': config_dict.get('ablation_EMA', True),
        'SR': config_dict.get('ablation_SR', True),
        'Adaptive': config_dict.get('ablation_adaptive', True),
        'AP': AP,
        'AF': AF,
        'AMF': AMF,
        'AG': AG,
        'AA': AA,
        'BWT': BWT,
        'ACC': ACC
    }


def main():
    """Main function for ablation study."""
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine which configs to run
    if args.ablation_config:
        # Run specific config
        configs_to_run = [(name, desc, cfg) for name, desc, cfg in ABLATION_CONFIGS 
                          if name == args.ablation_config]
        if not configs_to_run:
            print(f"Error: Unknown ablation config '{args.ablation_config}'")
            print(f"Available configs: {[c[0] for c in ABLATION_CONFIGS]}")
            return
    else:
        # Run all configs
        configs_to_run = ABLATION_CONFIGS
    
    print(f"\nAblation Study Configuration")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Backbone: {args.backbone_name}")
    print(f"Configs to run: {len(configs_to_run)}")
    print(f"Output: {output_dir}")
    
    # Store results
    results = []
    
    # Run experiments
    time_start = time.time()
    for idx, (config_name, config_desc, config_dict) in enumerate(configs_to_run):
        print(f"\n>>> Ablation {idx+1}/{len(configs_to_run)}: {config_name}")
        
        try:
            # Reset args for each experiment
            args.rst_min = 0.001
            args.rst_max = 0.01
            args.tao_begin = 0.95
            args.tao_end = 0.99
            
            result = run_single_ablation(args, config_name, config_desc, config_dict)
            results.append(result)
            
            # Save intermediate results
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = os.path.join(output_dir, f'ablation_results_intermediate_{timestamp}.csv')
            df.to_csv(intermediate_file, index=False)
            print(f"Intermediate results saved to: {intermediate_file}")
            
        except Exception as e:
            print(f"Error in ablation {config_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config_name': config_name,
                'config_desc': config_desc,
                'mixup': config_dict.get('ablation_mixup', True),
                'MI': config_dict.get('ablation_MI', True),
                'EMA': config_dict.get('ablation_EMA', True),
                'SR': config_dict.get('ablation_SR', True),
                'Adaptive': config_dict.get('ablation_adaptive', True),
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
    final_file = os.path.join(output_dir, f'ablation_results_final_{timestamp}.csv')
    df.to_csv(final_file, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {(time_end - time_start)/60:.2f} minutes")
    print(f"Results saved to: {final_file}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("ABLATION RESULTS COMPARISON")
    print(f"{'='*80}")
    cols_to_show = ['config_name', 'mixup', 'MI', 'EMA', 'SR', 'Adaptive', 'ACC', 'AP', 'AA', 'BWT']
    print(df[cols_to_show].to_string(index=False))
    
    # Compute contribution of each component
    print(f"\n{'='*80}")
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    
    full_result = df[df['config_name'] == 'Full']
    if not full_result.empty:
        full_acc = full_result['ACC'].values[0]
        for config_name in ['wo_Mixup', 'wo_MI', 'wo_EMA', 'wo_SR', 'wo_Adaptive']:
            wo_result = df[df['config_name'] == config_name]
            if not wo_result.empty:
                wo_acc = wo_result['ACC'].values[0]
                if wo_acc is not None and full_acc is not None:
                    contribution = full_acc - wo_acc
                    component = config_name.replace('wo_', '')
                    print(f"  {component}: {contribution:+.2f}% (Full: {full_acc:.2f}% -> w/o: {wo_acc:.2f}%)")
    
    return df


if __name__ == '__main__':
    main()
