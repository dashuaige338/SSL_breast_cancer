from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils_survival import *
from utils.core_utils_survival import train
from dataset_modules.dataset_generic_survival import Generic_Survival_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import re


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    # For large external-validation designs (for example, 28 leave-one-
    # institution-out folds), allow a previously selected alpha to be reused.
    # This performs one final training pass per fold instead of repeating the
    # four-alpha sweep inside every array task.
    if args.fixed_alpha is not None:
        args.alpha = float(args.fixed_alpha)
        fixed_val_cindex = []
        fixed_test_cindex = []
        print(f"Using fixed alpha={args.alpha}; skipping alpha sweep")
        with open(os.path.join(args.results_dir, "best_alpha.txt"), "w") as f:
            f.write(f"{args.alpha}\n")
        for i in folds:
            seed_torch(args.seed)
            train_dataset, val_dataset, test_dataset = dataset.return_splits(
                from_id=False,
                csv_path=os.path.join(args.split_dir, f'splits_{i}.csv')
            )
            val_cindex, test_cindex = train(
                (train_dataset, val_dataset, test_dataset), i, args
            )
            fixed_val_cindex.append(val_cindex)
            fixed_test_cindex.append(test_cindex)
        fixed_df = pd.DataFrame({
            'fold': folds,
            'valid_cindex': fixed_val_cindex,
            'test_cindex': fixed_test_cindex,
        })
        fixed_name = (
            f'summary_partial_{start}_{end}.csv' if len(folds) != args.k else 'summary.csv'
        )
        fixed_df.to_csv(os.path.join(args.results_dir, fixed_name), index=False)
        return fixed_df

    alpha_list = [1e-5, 1e-4, 1e-3, 1e-2]
    best_alpha = None
    best_mean_val_cindex = -1
    best_val_cindex = []
    best_test_cindex = []

    for alpha in alpha_list:
        print(f"Testing alpha={alpha}")
        args.alpha = alpha
        all_val_cindex = []
        all_test_cindex = []
        for i in folds:
            seed_torch(args.seed)
    
            train_dataset, val_dataset, test_dataset = dataset.return_splits(
                from_id=False, 
                csv_path=os.path.join(args.split_dir, f'splits_{i}.csv')
            )
            val_cindex, test_cindex = train((train_dataset, val_dataset, test_dataset), i, args)

            all_val_cindex.append(val_cindex)
            all_test_cindex.append(test_cindex)

        mean_val = np.mean(all_val_cindex)
        print(f"Alpha {alpha}: mean val c-index = {mean_val:.4f}")

        if mean_val > best_mean_val_cindex:
            best_mean_val_cindex = mean_val
            best_alpha = alpha
            best_val_cindex = all_val_cindex
            best_test_cindex = all_test_cindex

    print(f"\nBest alpha: {best_alpha} with mean val c-index = {best_mean_val_cindex:.4f}")

    # Save best alpha to file
    with open(os.path.join(args.results_dir, "best_alpha.txt"), "w") as f:
        f.write(f"{best_alpha}\n")

    # Retrain with best alpha on all folds to save the right checkpoints
    args.alpha = best_alpha
    for i in folds:
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, 
            csv_path=os.path.join(args.split_dir, f'splits_{i}.csv')
        )
        train((train_dataset, val_dataset, test_dataset), i, args)

    # Save summary
    final_df = pd.DataFrame({
        'fold': folds,
        'valid_cindex': best_val_cindex,
        'test_cindex': best_test_cindex,
    })

    summary_name = (
        f'summary_partial_{start}_{end}.csv' if len(folds) != args.k else 'summary.csv'
    )
    final_df.to_csv(os.path.join(args.results_dir, summary_name), index=False)


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_sb_surv'], default='clam_sb_surv', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, required = True, help='name of task (i.e. tcga_luad_survival_hoptimus_20x')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--fixed_alpha', type=float, default=None,
                    help='Skip the alpha sweep and train once using this Cox/L2 alpha.')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb', 'clam_sb_surv']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

# Replace task-based dataset logic with survival dataset initialization
print('\nLoad Dataset')

# Assume task is fixed for survival now
dataset = Generic_Survival_MIL_Dataset(
    csv_path=f'dataset_csv/{args.task}.csv',
    data_dir= os.path.join(args.data_root_dir, re.sub(r"(5x|10x|20x)", r"features_\1", args.task)),
    shuffle=False,
    seed=args.seed,
    print_info=True,
    patient_strat=True
)

args.n_classes = 1  # For Cox regression

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac * 100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")

