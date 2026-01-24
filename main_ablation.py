#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Bojian
Ablation study for ours_new method.

Component Hierarchy:
===================
Full Method = KD + CKCR + select_soft_knowledge + mixup + MI + EPHS + FISR + EMA

1. KD (Knowledge Distillation) - 基础，始终存在
   └── CKCR (Confidence-based Knowledge Consensus Regularization)
       ├── select_soft_knowledge (自适应熵门控，依赖CKCR)
       └── mixup (数据混合，依赖CKCR)
2. MI (Mutual Information loss) - 独立模块
3. EPHS (Entropy-aware Parameter Harmonization Scheduling) - 独立模块
   - 控制自适应 gate, rst, tao
4. EMA (Exponential Moving Average) - 始终运行，tao控制强度
5. FISR vs SR (互斥) - 随机恢复策略

Ablation Types:
- wo_X: 完整模型移除组件X (验证X的贡献)
- progressive: 从最小模型逐步添加组件
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
import copy

# ============ Dataset Domain Sequences ============
DATASET_DOMAIN_SEQS = {
    'SK': [
        np.array([6,1,8,15,22,17]),
        np.array([1,6,8,15,22,17]),
        np.array([6,8,1,15,22,17]),
        np.array([6,1,15,8,22,17]),
        np.array([6,1,8,22,15,17]),
        np.array([6,1,8,15,17,22])
    ],
    'iFlytek': [
        np.array([2,3,4,5,7]),
        np.array([3,2,4,5,7]),
        np.array([2,4,3,5,7]),
        np.array([2,3,5,4,7]),
        np.array([2,3,4,7,5])
    ],
    'WT': [
        np.array([0,1,2,3,4]),
        np.array([1,0,2,3,4]),
        np.array([0,2,1,3,4]),
        np.array([0,1,3,2,4]),
        np.array([0,1,2,4,3]),
    ],
        'PU_Real': [
        np.array([3,2,0,1]),
        np.array([2,3,0,1]),
        np.array([3,0,2,1]),
        np.array([3,2,1,0]),
    ],
    # 'PU_Art': [
    #     np.array([3,2,0,1]),
    #     np.array([2,3,0,1]),
    #     np.array([3,0,2,1]),
    #     np.array([3,2,1,0]),
    # ],
}

RANDOM_SEEDS = [2021,2022,2023,2024,2025]
# RANDOM_SEEDS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# ============ Full Model Configuration ============
FULL_CONFIG = {
    'CKCR': True,
    'select_soft_knowledge': True,
    'mixup': True,
    'MI': True,
    'EPHS': True,
    'FISR': True,
    'SR': False,
    'EMA': True,
    'Reset_student': True,
}

# ============ Ablation Configurations ============
# Format: (name, description, config_overrides)
ABLATION_CONFIGS = [
    # =============== 单组件移除 (Component-wise Ablation) ===============
    # 验证每个组件的独立贡献
    
    # 移除 CKCR (同时失去 select_soft_knowledge 和 mixup)
    ('wo_CKCR', 'Without CKCR (fallback to basic KD)', {
        'CKCR': False,
        'select_soft_knowledge': False,  # 依赖CKCR
        'mixup': False,  # 依赖CKCR
    }),
    
    # 移除 select_soft_knowledge (保留CKCR和mixup)
    ('wo_soft_knowledge', 'Without adaptive entropy gate', {
        'select_soft_knowledge': False,
    }),
    
    # 移除 mixup (保留CKCR和select_soft_knowledge)
    ('wo_mixup', 'Without mixup augmentation', {
        'mixup': False,
    }),
    
    # 移除 MI
    ('wo_MI', 'Without Mutual Information loss', {
        'MI': False,
    }),
    
    # 移除 EPHS (使用固定参数)
    ('wo_EPHS', 'Without adaptive parameter scheduling', {
        'EPHS': False,
    }),
    
    # 移除 FISR (用基础SR替代)
    ('wo_FISR', 'Without Fisher-weighted SR (use basic SR instead)', {
        'FISR': False,
        'SR': True,
    }),
    
    # 移除所有SR (FISR和SR都不用)
    ('wo_allSR', 'Without any Stochastic Recovery', {
        'FISR': False,
        'SR': False,
    }),
    
    # =============== Mean-Teacher 相关消融 ===============
    # 移除 EMA (不使用 Mean-Teacher 框架，直接用 student 更新)
    ('wo_EMA', 'Without EMA (no Mean-Teacher, student-only updates)', {
        'EMA': False,
        'CKCR': False,
    }),
    
    # 移除 Reset_student (不在每个 epoch 后用 teacher 重置 student)
    ('wo_Reset_student', 'Without resetting student from teacher each epoch', {
        'Reset_student': False,
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

### Ablation flags (used by ours_new.py)
parser.add_argument('--CKCR', action='store_true', help='Enable CKCR')
parser.add_argument('--select_soft_knowledge', action='store_true', help='Enable soft knowledge selection')
parser.add_argument('--mixup', action='store_true', help='Enable mixup')
parser.add_argument('--EPHS', action='store_true', help='Enable EPHS')
parser.add_argument('--FISR', action='store_true', help='Enable Fisher-weighted SR')
parser.add_argument('--SR', action='store_true', help='Enable basic SR')
parser.add_argument('--MI', action='store_true', help='Enable MI loss')
parser.add_argument('--EMA', action='store_true', help='Enable EMA for Mean-Teacher')
parser.add_argument('--Reset_student', action='store_true', help='Reset student from teacher each epoch')
parser.add_argument('--TOPK', action='store_false', help='the ablation setting')
parser.add_argument('--PCA', action='store_false', help='the ablation setting')

### Hyperparameters
parser.add_argument('--rst_min', default=0.001, type=float, help='rst_min for SR')
parser.add_argument('--rst_max', default=0.01, type=float, help='rst_max for SR')
parser.add_argument('--tao_begin', default=0.95, type=float, help='tao_begin for EMA')
parser.add_argument('--tao_end', default=0.99, type=float, help='tao_end for EMA')

### Ablation experiment settings
parser.add_argument('--ablation_config', default=None, type=str, help='Specific ablation config to run')
parser.add_argument('--output_dir', default='./log/ablation/', type=str, help='Output directory')
parser.add_argument('--run_all_datasets', action='store_true', help='Run on all 3 datasets')
parser.add_argument('--run_multi_seed', action='store_true', help='Run with multiple seeds')
parser.add_argument('--run_multi_domain', action='store_true', help='Run with multiple domain sequences')

args = parser.parse_args()


def setup_dataset_args(args, dataset_name, domain_seq=None):
    """Setup dataset-specific arguments."""
    args.dataset_name = dataset_name
    
    if dataset_name == 'SK':
        args.train_list = './SK_all_10classes.mat'
        args.test_list = './SK_all_10classes.mat'
        args.Domain_Seq = domain_seq if domain_seq is not None else np.array([6,1,8,15,22,17])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 10
        args.batch_size = 64
        # Grid-search optimized hyperparameters for SK
        args.tao_begin = 0.95
        args.tao_end = 0.99
        args.rst_min = 0.005
        args.rst_max = 0.05
        
    elif dataset_name == 'iFlytek':
        args.train_list = './iFlytek_all_5classes.mat'
        args.test_list = './iFlytek_all_5classes.mat'
        args.Domain_Seq = domain_seq if domain_seq is not None else np.array([2,3,4,5,7])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5
        args.batch_size = 64
        # Grid-search optimized hyperparameters for iFlytek
        args.tao_begin = 0.95
        args.tao_end = 0.97
        args.rst_min = 0.0005
        args.rst_max = 0.1
        
    elif dataset_name == 'WT':
        args.train_list = './WT_all_5classes.mat'
        args.test_list = './WT_all_5classes.mat'
        args.Domain_Seq = domain_seq if domain_seq is not None else np.array([0,1,2,3,4])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5
        args.batch_size = 128
        # Grid-search optimized hyperparameters for WT
        args.tao_begin = 0.95
        args.tao_end = 0.97
        args.rst_min = 0.01
        args.rst_max = 0.05
    
    # Set model-specific args
    if args.backbone_name == 'cnn' or args.backbone_name == 'resnet18_1D':
        args.data_dimension = '1D'
        args.data_mode = 'Time'
    if args.backbone_name == 'resnet14' or args.backbone_name == 'resnet32':
        args.data_dimension = '2D'
        args.data_mode = 'Frequence'
    
    return args


def build_config(overrides):
    """Build full config by applying overrides to FULL_CONFIG."""
    config = FULL_CONFIG.copy()
    config.update(overrides)
    return config


def apply_config_to_args(args, config):
    """Apply configuration dict to args."""
    args.CKCR = config['CKCR']
    args.select_soft_knowledge = config['select_soft_knowledge']
    args.mixup = config['mixup']
    args.MI = config['MI']
    args.EPHS = config['EPHS']
    args.FISR = config['FISR']
    args.SR = config['SR']
    args.EMA = config['EMA']
    args.Reset_student = config['Reset_student']
    return args


def run_single_experiment(args, config_name, config, dataset_name, seed, domain_seq):
    """Run a single ablation experiment."""
    # Setup
    args = setup_dataset_args(args, dataset_name, domain_seq)
    args.random_seed = seed
    set_random_seed(seed)
    args = apply_config_to_args(args, config)
    
    print(f"\n{'='*80}")
    print(f"Config: {config_name} | Dataset: {dataset_name} | Seed: {seed}")
    print(f"Domain_Seq: {domain_seq}")
    print(f"CKCR={args.CKCR}, soft={args.select_soft_knowledge}, mixup={args.mixup}, "
          f"MI={args.MI}, EPHS={args.EPHS}, FISR={args.FISR}, SR={args.SR}")
    print(f"{'='*80}")
    
    # Run training
    Correct = train(args)
    
    # Compute metrics
    AP, AF, AMF, AG, AA, BWT, ACC = eval_metric(args, Correct)
    
    return {
        'config_name': config_name,
        'dataset': dataset_name,
        'seed': seed,
        'domain_seq': str(domain_seq.tolist()),
        'CKCR': config['CKCR'],
        'select_soft_knowledge': config['select_soft_knowledge'],
        'mixup': config['mixup'],
        'MI': config['MI'],
        'EPHS': config['EPHS'],
        'FISR': config['FISR'],
        'SR': config['SR'],
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
    
    # Determine datasets to run
    if args.run_all_datasets:
        datasets = ['SK', 'iFlytek', 'WT']
    else:
        datasets = [args.dataset_name]
    
    # Create output directory per dataset
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine seeds
    if args.run_multi_seed:
        seeds = RANDOM_SEEDS
    else:
        seeds = [args.random_seed]
    
    # Determine configs to run
    if args.ablation_config:
        configs_to_run = [(name, desc, build_config(overrides)) 
                          for name, desc, overrides in ABLATION_CONFIGS 
                          if name == args.ablation_config]
        if not configs_to_run:
            print(f"Error: Unknown ablation config '{args.ablation_config}'")
            print(f"Available configs: {[c[0] for c in ABLATION_CONFIGS]}")
            return
    else:
        configs_to_run = [(name, desc, build_config(overrides)) 
                          for name, desc, overrides in ABLATION_CONFIGS]
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY CONFIGURATION")
    print(f"{'='*80}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Multi-domain: {args.run_multi_domain}")
    print(f"Configs: {[c[0] for c in configs_to_run]}")
    print(f"Output: {output_dir}")
    
    # Calculate total experiments
    total_exps = 0
    for dataset in datasets:
        n_domain = len(DATASET_DOMAIN_SEQS.get(dataset, [None])) if args.run_multi_domain else 1
        total_exps += len(configs_to_run) * len(seeds) * n_domain
    print(f"Total experiments: {total_exps}")
    
    # Store results
    results = []
    exp_count = 0
    
    time_start = time.time()
    
    for config_name, config_desc, config in configs_to_run:
        for dataset in datasets:
            # Get domain sequences for this dataset
            if args.run_multi_domain and dataset in DATASET_DOMAIN_SEQS:
                domain_seqs = DATASET_DOMAIN_SEQS[dataset]
            else:
                domain_seqs = [DATASET_DOMAIN_SEQS[dataset][0] if dataset in DATASET_DOMAIN_SEQS else None]
            
            for domain_seq in domain_seqs:
                for seed in seeds:
                    exp_count += 1
                    print(f"\n>>> Experiment {exp_count}/{total_exps}")
                    
                    try:
                        # Create fresh args copy
                        exp_args = copy.deepcopy(args)
                        result = run_single_experiment(exp_args, config_name, config, 
                                                       dataset, seed, domain_seq)
                        results.append(result)
                        
                        # Save intermediate results
                        df = pd.DataFrame(results)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        intermediate_file = os.path.join(output_dir, f'ablation_intermediate_{timestamp}.csv')
                        df.to_csv(intermediate_file, index=False)
                        
                    except Exception as e:
                        print(f"Error: {e}")
                        import traceback
                        traceback.print_exc()
                        results.append({
                            'config_name': config_name,
                            'dataset': dataset,
                            'seed': seed,
                            'domain_seq': str(domain_seq.tolist()) if domain_seq is not None else 'None',
                            'error': str(e)
                        })
    
    time_end = time.time()
    
    # Save final results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = os.path.join(output_dir, f'ablation_final_{timestamp}.csv')
    df.to_csv(final_file, index=False)
    
    # Compute and save mean results
    print(f"\n{'='*80}")
    print("COMPUTING MEAN RESULTS")
    print(f"{'='*80}")
    
    metrics = ['AP', 'AF', 'AMF', 'AG', 'AA', 'BWT', 'ACC']
    group_cols = ['config_name', 'dataset']
    
    # Filter valid results
    valid_df = df[df['ACC'].notna()].copy()
    
    if not valid_df.empty:
        # Compute mean and std
        mean_df = valid_df.groupby(group_cols)[metrics].mean().reset_index()
        std_df = valid_df.groupby(group_cols)[metrics].std().reset_index()
        
        # Rename columns
        mean_df.columns = group_cols + [f'{m}_mean' for m in metrics]
        std_df.columns = group_cols + [f'{m}_std' for m in metrics]
        
        # Merge
        summary_df = pd.merge(mean_df, std_df, on=group_cols)
        
        # Save summary
        summary_file = os.path.join(output_dir, f'ablation_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nSummary saved to: {summary_file}")
        print(summary_df[['config_name', 'dataset', 'ACC_mean', 'ACC_std', 'AP_mean', 'BWT_mean']].to_string(index=False))
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {(time_end - time_start)/60:.2f} minutes")
    print(f"Results: {final_file}")


if __name__ == '__main__':
    main()
