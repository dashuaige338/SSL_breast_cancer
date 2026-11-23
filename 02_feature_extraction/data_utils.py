from glob import glob
import numpy as np
import pandas as pd
import os

def concat_slide_embeddings(model):
    """Concatenate slide embeddings stored in separate csvs, adds a sample column."""
    csv_files = glob(f"/gpfs/data/abl/deepomics/meskerlab/breast_model_embeddings/{model}/*/*{model}_slide_features.csv")
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        df = df.iloc[:, 1:]
        sample_name = os.path.basename(file).split("_")[0]
        df["samples"] = sample_name 
        dataframes.append(df)
    final_df = pd.concat(dataframes, ignore_index=True)
    return final_df