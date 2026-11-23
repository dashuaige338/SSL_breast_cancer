import argparse
import os
import glob
import time
from utils import *
from PathologyFoundationModels.run_virchow_widerer_v2 import *
from PathologyFoundationModels.run_uni_widerer_v15 import *
from PathologyFoundationModels.run_hoptimusv0 import *
from PathologyFoundationModels.run_keep import *
from PathologyFoundationModels.run_titan import *

parser = argparse.ArgumentParser(description="To select the pathology fundation models and save the generated embeddings to the output path")

parser.add_argument("--input_path", type=str, required=True, help="Path to your svs/tiff files")
parser.add_argument("--output_path", type=str, required=True, help="Path you want to save your embeddings")
parser.add_argument("--model", type=str, required=True, help="The pathology fundation models you want to use")

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path
model = args.model

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    all_slides=sorted(glob.glob(f'{input_path}/*.mrxs'))
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
        except ValueError as e:
            print(e)
            print("No slide found")
            continue
        print("Complete patch extraction!")
        if model=='TITAN':
            print("Running TITAN")
            try:
                conch_tile_embeddings, titan_slide_embeddings = run_titan(['LUAD'],patch_gen, patient_names[i], output_path)
            except Exception as e:
                print(f"Slide {all_slides[i]} failed: {e}")
                continue
        if model=='Hoptimusv0':
            print("Running hoptimus_v0")
            try:
                hoptimus_embeddings = run_hoptimus_v0(patch_gen, patient_names[i], output_path)
            except Exception as e:
                print(f"Slide {all_slides[i]} failed: {e}")
                continue
        if model=='Virchow':
            print("Running Virchow-V2")
            try:
                virchow_v2_embeddings = run_virchow_v2(patch_gen, patient_names[i], output_path)
            except Exception as e:
                print(f"Slide {all_slides[i]} failed: {e}")
                continue
        if model=='UNI':
            print("Running UNI-V2")
            try:
                univ2_embeddings = run_uni_v2(patch_gen, patient_names[i], output_path)
            except Exception as e:
                print(f"Slide {all_slides[i]} failed: {e}")
                continue  
        if model=='PROVGIGAPATH':
            from PathologyFoundationModels.run_prov_gigapath import *
            print("Running PROVGIGAPATH")
            try:
                tile_embeddings, slide_embeddings = run_prov_gigapath(patch_gen, patient_names[i], output_path)
            except Exception as e:
                print(f"Slide {all_slides[i]} failed: {e}")
                continue  
        if model=='KEEP':
            print("Running KEEP")
            try:
                all_matchings = run_keep(['invasive ductal carcinoma', 'invasive lobular carcinoma'],patch_gen, patient_names[i], output_path)
            except Exception as e:
                print(f"Slide {all_slides[i]} failed: {e}")
                continue



            