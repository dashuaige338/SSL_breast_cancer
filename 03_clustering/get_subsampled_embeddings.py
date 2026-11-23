import numpy as np
import pandas as pd
import pickle
import os
import tqdm
from glob import glob

def read_embeddings_coords(path):
    embedding_df = pd.read_csv(path)
    embedding_df = embedding_df.iloc[:, 1:]
    name = path.split('/')[-1].split('.')[0].split('_')[0]
    coord_df = pd.read_csv(f'/gpfs/data/abl/deepomics/meskerlab/breast_model_embeddings/coords/{name}_coords.csv')
    df=pd.concat([embedding_df,coord_df[['x','y']]],axis=1)
    return df

with open('/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/kmeans/subsampled_10K.pkl', 'rb') as f:
    subsets_10k = pickle.load(f)

with open('/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/kmeans/subsampled_100K.pkl', 'rb') as f:
    subsets_100k = pickle.load(f)

emb_csv_files = glob(f"/gpfs/data/abl/deepomics/meskerlab/breast_model_embeddings/Virchow/*/*csv")

seeds = [42, 0, 12345, 2023, 99]
subsets_100k_embs = {}
for seed in seeds:
    df_sample = subsets_100k[seed]
    patients = list(set(df_sample['patient'].tolist()))
    dfs_all = []
    for patient in tqdm.tqdm(patients):
        patient_df_sample = df_sample[df_sample['patient']==patient]
        path = next(s for s in emb_csv_files if patient in s)
        emb_df = read_embeddings_coords(path)
        keys = patient_df_sample[['x', 'y']].drop_duplicates()
        emb_df_sample = emb_df.merge(keys, on=['x', 'y'], how='inner')
        dfs_all.append(emb_df_sample)
    df_all=pd.concat(dfs_all)
    subsets_100k_embs[seed]=df_all 

with open('/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/kmeans/Virchow/subsampled_100K_embeddings.pkl', 'wb') as f:
    pickle.dump(subsets_100k_embs, f)