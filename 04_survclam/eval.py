from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
from utils.utils_survival import *
from math import floor
from dataset_modules.dataset_generic_survival import Generic_Survival_MIL_Dataset, save_splits
import h5py
import pickle
from utils.eval_utils_survival import *
import re

# Training settings
parser = argparse.ArgumentParser(description='CLAM Survival Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_sb_surv'], default='clam_sb_surv')
parser.add_argument('--k', type=int, default=10, help='number of folds')
parser.add_argument('--k_start', type=int, default=-1)
parser.add_argument('--k_end', type=int, default=-1)
parser.add_argument('--fold', type=int, default=-1)
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--drop_out', type=float, default=0.25)
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--task', type=str, required = True, help='name of task (i.e. tcga_luad_survival_hoptimus_20x')
parser.add_argument('--predictions_only', action='store_true', default=False,
                    help='Save fold predictions/metrics without large embedding and attention files.')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)

print(settings)

# Load dataset for survival
dataset = Generic_Survival_MIL_Dataset(
    csv_path=f'dataset_csv/{args.task}.csv',
    data_dir= os.path.join(args.data_root_dir, re.sub(r"(5x|10x|20x)", r"features_\1", args.task)),
    shuffle=False,
    print_info=True,
    patient_strat=False)

# Cox regression
args.n_classes = 1

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)

ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_cindex = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        '''
        model, cindex, df = eval_survival(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(df)
        all_cindex.append(cindex)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)'''
        model, cindex, df, embeddings, attention, attention_raw = eval_survival(
            split_dataset, args, ckpt_paths[ckpt_idx]
            )

        fold_id = folds[ckpt_idx]

        all_results.append(df)
        all_cindex.append(cindex)

        # Save prediction dataframe
        df.to_csv(
            os.path.join(args.save_dir, 'fold_{}.csv'.format(fold_id)),
            index=False
        )

        # Save large interpretability artifacts only when explicitly wanted.
        if embeddings is not None and not args.predictions_only:
            np.save(
                os.path.join(args.save_dir, 'fold_{}_embeddings.npy'.format(fold_id)),
                embeddings
            )

        # Save attention scores as pickle because patch number differs by slide
        if not args.predictions_only:
            with open(os.path.join(args.save_dir, 'fold_{}_attention.pkl'.format(fold_id)), 'wb') as f:
                pickle.dump(attention, f)

            with open(os.path.join(args.save_dir, 'fold_{}_attention_raw.pkl'.format(fold_id)), 'wb') as f:
                pickle.dump(attention_raw, f)

    final_df = pd.DataFrame({'folds': list(folds), 'cindex': all_cindex})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
