#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Chen Bojian
Parallel experiment runner for ours_new method.
Supports running multiple experiments in parallel across different seeds, datasets, and domain sequences.
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
from multiprocessing import Pool, cpu_count
import traceback

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
    # 'PU_Real': [
    #     np.array([3,2,0,1]),
    #     np.array([2,3,0,1]),
    #     np.array([3,0,2,1]),
    #     np.array([3,2,1,0]),
    # ],
    # 'PU_Art': [
    #     np.array([3,2,0,1]),
    #     np.array([2,3,0,1]),
    #     np.array([3,0,2,1]),
    #     np.array([3,2,1,0]),
    # ],
}

RANDOM_SEEDS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

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


def create_parser():
    """Create argument parser."""
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

    ### ours_new ablation flags
    parser.add_argument('--CKCR', action='store_false', help='Enable CKCR')
    parser.add_argument('--select_soft_knowledge', action='store_false', help='Enable soft knowledge selection')
    parser.add_argument('--mixup', action='store_false', help='Enable mixup')
    parser.add_argument('--EPHS', action='store_false', help='Enable EPHS')
    parser.add_argument('--FISR', action='store_false', help='Enable Fisher-weighted SR')
    parser.add_argument('--SR', action='store_false', help='Enable basic SR')
    parser.add_argument('--MI', action='store_false', help='Enable MI loss')
    parser.add_argument('--EMA', action='store_false', help='Enable EMA for Mean-Teacher')
    parser.add_argument('--Reset_student', action='store_false', help='Reset student from teacher each epoch')
    parser.add_argument('--TOPK', action='store_false', help='the ablation setting')
    parser.add_argument('--PCA', action='store_false', help='the ablation setting')

    ### Hyperparameters
    parser.add_argument('--rst_min', default=0.001, type=float, help='rst_min for SR')
    parser.add_argument('--rst_max', default=0.01, type=float, help='rst_max for SR')
    parser.add_argument('--tao_begin', default=0.95, type=float, help='tao_begin for EMA')
    parser.add_argument('--tao_end', default=0.99, type=float, help='tao_end for EMA')

    ### Parallel experiment settings
    parser.add_argument('--output_dir', default='./log/parallel/', type=str, help='Output directory')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of parallel workers (1=sequential)')
    parser.add_argument('--run_all_datasets', action='store_true', help='Run on all datasets')
    parser.add_argument('--run_multi_seed', action='store_true', help='Run with multiple seeds')
    parser.add_argument('--run_multi_domain', action='store_true', help='Run with multiple domain sequences')
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU IDs to use, comma separated (e.g., "0,1,2")')

    return parser



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
        args.tao_begin = 0.97
        args.tao_end = 0.99
        args.rst_min = 0.0005
        args.rst_max = 0.01
    
    elif dataset_name == 'SK_new':
        args.train_list = './SK_new_all_10classes.mat'
        args.test_list = './SK_new_all_10classes.mat'
        args.Domain_Seq = domain_seq if domain_seq is not None else np.array([6,1,8,15,22,17])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 10
        args.batch_size = 64
        
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
    
    elif dataset_name == 'PU_Real':
        args.train_list = './PU_Real_4doamins_5classes.mat'
        args.test_list = './PU_Real_4doamins_5classes.mat'
        args.Domain_Seq = domain_seq if domain_seq is not None else np.array([3,2,0,1])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 5
        args.batch_size = 64
    
    elif dataset_name == 'PU_Art':
        args.train_list = './PU_Art_4doamins_8classes.mat'
        args.test_list = './PU_Art_4doamins_8classes.mat'
        args.Domain_Seq = domain_seq if domain_seq is not None else np.array([2,0])
        args.nb_session = len(args.Domain_Seq)
        args.nb_cl = 8
        args.batch_size = 64
    
    # Set model-specific args
    if args.backbone_name == 'cnn' or args.backbone_name == 'resnet18_1D':
        args.data_dimension = '1D'
        args.data_mode = 'Time'
    if args.backbone_name == 'resnet14' or args.backbone_name == 'resnet32':
        args.data_dimension = '2D'
        args.data_mode = 'Frequence'
    
    return args


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


def run_single_experiment(exp_config):
    """Run a single experiment. Designed for multiprocessing.
    
    Args:
        exp_config: Tuple of (args_dict, dataset_name, seed, domain_seq, gpu_id, exp_id, total_exps)
    
    Returns:
        Dictionary with experiment results
    """
    args_dict, dataset_name, seed, domain_seq, gpu_id, exp_id, total_exps = exp_config
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Recreate args from dict
    parser = create_parser()
    args = parser.parse_args([])
    for key, value in args_dict.items():
        setattr(args, key, value)
    
    try:
        # Setup
        args = setup_dataset_args(args, dataset_name, domain_seq)
        args.random_seed = seed
        set_random_seed(seed)
        args = apply_config_to_args(args, FULL_CONFIG)
        
        print(f"\n{'='*80}")
        print(f"[{exp_id}/{total_exps}] Dataset: {dataset_name} | Seed: {seed} | GPU: {gpu_id}")
        print(f"Domain_Seq: {domain_seq}")
        print(f"CKCR={args.CKCR}, soft={args.select_soft_knowledge}, mixup={args.mixup}, "
              f"MI={args.MI}, EPHS={args.EPHS}, FISR={args.FISR}, SR={args.SR}")
        print(f"{'='*80}")
        
        # Run training
        Correct = train(args)
        
        # Compute metrics
        AP, AF, AMF, AG, AA, BWT, ACC = eval_metric(args, Correct)
        
        return {
            'exp_id': exp_id,
            'dataset': dataset_name,
            'seed': seed,
            'domain_seq': str(domain_seq.tolist()),
            'gpu_id': gpu_id,
            'CKCR': args.CKCR,
            'select_soft_knowledge': args.select_soft_knowledge,
            'mixup': args.mixup,
            'MI': args.MI,
            'EPHS': args.EPHS,
            'FISR': args.FISR,
            'SR': args.SR,
            'AP': AP,
            'AF': AF,
            'AMF': AMF,
            'AG': AG,
            'AA': AA,
            'BWT': BWT,
            'ACC': ACC,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Error in experiment {exp_id}: {e}")
        traceback.print_exc()
        return {
            'exp_id': exp_id,
            'dataset': dataset_name,
            'seed': seed,
            'domain_seq': str(domain_seq.tolist()) if domain_seq is not None else 'None',
            'gpu_id': gpu_id,
            'error': str(e),
            'status': 'failed'
        }


def main():
    """Main function for parallel experiments."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(',')]
    
    # Determine datasets to run
    if args.run_all_datasets:
        datasets = ['SK', 'iFlytek', 'WT', 'PU_Real', 'PU_Art']
    else:
        datasets = [args.dataset_name]
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine seeds
    if args.run_multi_seed:
        seeds = RANDOM_SEEDS
    else:
        seeds = [args.random_seed]
    
    # Generate all experiment configurations
    exp_configs = []
    exp_id = 0
    
    # Convert args to dict for multiprocessing
    args_dict = vars(args).copy()
    
    for dataset in datasets:
        # Get domain sequences for this dataset
        if args.run_multi_domain and dataset in DATASET_DOMAIN_SEQS:
            domain_seqs = DATASET_DOMAIN_SEQS[dataset]
        else:
            domain_seqs = [DATASET_DOMAIN_SEQS[dataset][0] if dataset in DATASET_DOMAIN_SEQS else None]
        
        for domain_seq in domain_seqs:
            for seed in seeds:
                exp_id += 1
                gpu_id = gpu_ids[(exp_id - 1) % len(gpu_ids)]
                exp_configs.append((args_dict, dataset, seed, domain_seq, gpu_id, exp_id, -1))
    
    # Update total experiment count
    total_exps = len(exp_configs)
    exp_configs = [(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], total_exps) for cfg in exp_configs]
    
    print(f"\n{'='*80}")
    print("PARALLEL EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Multi-domain: {args.run_multi_domain}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Num workers: {args.num_workers}")
    print(f"Total experiments: {total_exps}")
    print(f"Output: {output_dir}")
    
    # Run experiments
    results = []
    time_start = time.time()
    
    if args.num_workers == 1:
        # Sequential execution
        for exp_config in exp_configs:
            result = run_single_experiment(exp_config)
            results.append(result)
            
            # Save intermediate results
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = os.path.join(output_dir, f'parallel_intermediate_{timestamp}.csv')
            df.to_csv(intermediate_file, index=False)
    else:
        # Parallel execution
        with Pool(processes=args.num_workers) as pool:
            for result in pool.imap_unordered(run_single_experiment, exp_configs):
                results.append(result)
                
                # Save intermediate results
                df = pd.DataFrame(results)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                intermediate_file = os.path.join(output_dir, f'parallel_intermediate_{timestamp}.csv')
                df.to_csv(intermediate_file, index=False)
    
    time_end = time.time()
    
    # Save final results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = os.path.join(output_dir, f'parallel_final_{timestamp}.csv')
    df.to_csv(final_file, index=False)
    
    # Compute and save summary
    print(f"\n{'='*80}")
    print("COMPUTING SUMMARY")
    print(f"{'='*80}")
    
    metrics = ['AP', 'AF', 'AMF', 'AG', 'AA', 'BWT', 'ACC']
    group_cols = ['dataset']
    
    # Filter valid results
    valid_df = df[df['status'] == 'success'].copy()
    
    if not valid_df.empty:
        # Compute mean and std
        mean_df = valid_df.groupby(group_cols)[metrics].mean().reset_index()
        std_df = valid_df.groupby(group_cols)[metrics].std().reset_index()
        count_df = valid_df.groupby(group_cols)[metrics].count().reset_index()
        
        # Rename columns
        mean_df.columns = group_cols + [f'{m}_mean' for m in metrics]
        std_df.columns = group_cols + [f'{m}_std' for m in metrics]
        count_df.columns = group_cols + [f'{m}_count' for m in metrics]
        
        # Merge
        summary_df = pd.merge(mean_df, std_df, on=group_cols)
        summary_df = pd.merge(summary_df, count_df[['dataset', 'ACC_count']], on=group_cols)
        
        # Save summary
        summary_file = os.path.join(output_dir, f'parallel_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nSummary saved to: {summary_file}")
        print(summary_df[['dataset', 'ACC_mean', 'ACC_std', 'AP_mean', 'BWT_mean', 'ACC_count']].to_string(index=False))
    
    # Print failed experiments
    failed_df = df[df['status'] == 'failed']
    if not failed_df.empty:
        print(f"\n{'='*80}")
        print(f"FAILED EXPERIMENTS: {len(failed_df)}")
        print(f"{'='*80}")
        print(failed_df[['exp_id', 'dataset', 'seed', 'error']].to_string(index=False))
    
    print(f"\n{'='*80}")
    print("PARALLEL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {(time_end - time_start)/60:.2f} minutes")
    print(f"Successful: {len(valid_df)}/{total_exps}")
    print(f"Results: {final_file}")


if __name__ == '__main__':
    main()
