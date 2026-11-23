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
from glob import glob
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="To select the pathology fundation models and save the generated kmeans label to the output path")
parser.add_argument("--model", type=str, required=True, help="The pathology fundation models you want to use")
#parser.add_argument("--clusters", type=str, required=True, help="The number of the k-means clusters you want")
parser.add_argument("--dimensions", type=int, required=True, help="The dimensions of your fundation models")
args = parser.parse_args()
model = args.model
#clusters = args.clusters
n_dimensions = args.dimensions
'''
def read_embeddings_coords(path):
    embedding_df = pd.read_csv(path)
    embedding_df = embedding_df.iloc[:, 1:]
    name = path.split('/')[-1].split('.')[0].split('_')[0]
    coord_df = pd.read_csv(f'/gpfs/data/abl/deepomics/meskerlab/breast_model_embeddings/coords/{name}_coords.csv')
    df=pd.concat([embedding_df,coord_df[['x','y']]],axis=1)
    return df'''


if __name__ == "__main__":
    '''csv_files = glob(f"/gpfs/data/abl/deepomics/meskerlab/breast_model_embeddings/{model}/*/*csv")

    dfs = []
    for path in csv_files:
        df = read_embeddings_coords(path)
        expe = (len(df)*1000000)/16517645
        num = random.randint(0, round(2*expe))
        if num != 0:
            df_sample = df.sample(n=num)
        dfs.append(df_sample)
    df_all=pd.concat(dfs,ignore_index=True)
    print(len(df_all))'''
    df_all = pd.read_parquet(f"/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/kmeans/{model}_subsampled_1M.parquet", engine="pyarrow")
    X = df_all.iloc[:, :n_dimensions].copy()
    out_path = f'/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/kmeans/{model}'
    os.makedirs(out_path, exist_ok=True)
    results = pd.DataFrame(columns=['clusters', 'sil', 'ch', 'db'])
    for n in range(20,101,10):
        kmeans = KMeans(n_clusters=n, n_init=20, random_state=42)
        #kmeans.fit(X)
        labels = kmeans.fit_predict(X)
        with open(f'{out_path}/kmeans_model_{n}.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        sil = silhouette_score(X, labels, metric='euclidean')
        ch  = calinski_harabasz_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        results.loc[len(results)] = [n, sil, ch, db]
    results.to_csv(f'{out_path}/metrics.csv',index=False)
