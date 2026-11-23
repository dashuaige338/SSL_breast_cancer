import argparse
import os
import glob
import time
from utils import *

parser = argparse.ArgumentParser(description="To select the pathology fundation models and save the generated embeddings to the output path")

parser.add_argument("--input_path", type=str, required=True, help="Path to your svs/tiff files")
parser.add_argument("--output_path", type=str, required=True, help="Path you want to save your embeddings")

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    all_slides=sorted(glob.glob(f'{input_path}/*/*.mrxs'))
    patient_names=[i.split('/')[-1].split('.')[0] for i in all_slides]
    for i in range(len(all_slides)):
        print(i)
        name = patient_names[i]
        existing = glob.glob(os.path.join(output_path, f'{name}*'))
        if existing:
            print(f"output for patient '{name}' already exists.")
            continue
        print(f'Start working with {all_slides[i]}')
        try:
            patch_gen=get_existing_slide(all_slides[i])
            patch_gen.locations_df.to_csv(f'{output_path}/{name}_coords.csv',index=False)
        except ValueError as e:
            print(e)
            print("No slide found")
            continue
        print("Complete patch extraction!")
        