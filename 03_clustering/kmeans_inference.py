import numpy as np
import pandas as pd
import scanpy as sc
import fastcluster
import anndata
import h5py
import sys
from pandas.api.types import is_numeric_dtype
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import scipy.stats as st
import random
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import spearmanr
from lifelines.plotting import add_at_risk_counts
from scipy.stats import ttest_ind
import glob
from sklearn.cluster import KMeans
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="To select the pathology fundation models and save the generated embeddings to the output path")


parser.add_argument("--input_path", type=str, required=True, help="Path to your svs/tiff files")
parser.add_argument("--output_path", type=str, required=True, help="Path you want to save your embeddings")
parser.add_argument("--model", type=str, required=True, help="The pathology fundation models you want to use")
parser.add_argument("--dimensions", type=int, required=True, help="The dimensions of your fundation models")

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path
model = args.model
n_dimensions = args.dimensions

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    all_slides=sorted(glob.glob(f'{input_path}/{model}/*/*.csv'))
    patient_names=[i.split('/')[-1].split('.')[0].split('_')[0] for i in all_slides]
    with open(f'/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/kmeans/{model}/kmeans_model_20_test.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    for i in range(len(all_slides)):
        df = pd.read_csv(all_slides[i])
        X = df.iloc[:, 1:n_dimensions+1].copy()
        labels = kmeans.predict(X)
        label_df = pd.DataFrame({'cluster':labels,'patient':df['patient'].tolist()})
        label_df.to_csv(f'{output_path}/{patient_names[i]}_{model}_kmeans_labels.csv',index=False)

