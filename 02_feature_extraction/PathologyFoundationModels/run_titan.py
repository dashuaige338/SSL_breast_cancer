#from conch.open_clip_custom import create_model_from_pretrained #pip install git+https://github.com/Mahmoodlab/CONCH.git
import os
import torch
from huggingface_hub import login
from transformers import AutoModel 
import tifffile
from torchvision import transforms #type:ignore
#import h5py
from transformers import AutoModel
#from conch.open_clip_custom import create_model_from_pretrained
import requests  # Used for making HTTP requests.
import os  # Provides file and directory operations.
import torch  # Pytorch library for machine learning.
# For semantic segmentation
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor  # For WSI segmentation.
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils import imwrite
import numpy as np
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.models import IOSegmentorConfig
import pandas as pd #type: ignore
from tiatoolbox import rcParam #type: ignore
from pathlib import Path
#Now, let's create a KEEP run using those prompts.
from transformers import AutoModel, AutoTokenizer, pipeline
from torchvision import transforms
from PIL import Image
from huggingface_hub import login #type:ignore
import cv2
import timm  #type:ignore
from timm.data import resolve_data_config #type: ignore
from timm.data.transforms_factory import create_transform  #type: ignore
from timm.layers import SwiGLUPacked  #type: ignore
import tifffile #type: ignore
#import pyvips #type: ignore
import time
from PIL.TiffTags import TAGS #type: ignore
import PIL #type: ignore
import sys
import re
#from conch.open_clip_custom import create_model_from_pretrained #pip install git+https://github.com/Mahmoodlab/CONCH.git
from transformers import AutoModel 

TEMPLATEs = [
    "CLASSNAME.",
    "an image of CLASSNAME.",
    "the image shows CLASSNAME.",
    "the image displays CLASSNAME.",
    "the image exhibits CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "I observe CLASSNAME.",
    "the pathology image shows CLASSNAME.",
    "a pathology image shows CLASSNAME.",
    "the pathology slide shows CLASSNAME.",
    "shows CLASSNAME.",
    "contains CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "CLASSNAME is observed.",
    "the pathology image reveals CLASSNAME.",
    "a microscopic image of showing CLASSNAME.",
    "histology shows CLASSNAME.",
    "CLASSNAME can be seen.",
    "the tissue shows CLASSNAME.",
    "CLASSNAME is identified.",
]

def run_titan(disease_classification, patch_gen, patient_name, output_path, other_prompts = None,factor_use=1):
    """
    Runs the TITAN model to match text prompts with image patches.
    
    TODO: Fix docstring
    TODO: Text processing with encodings on the YAML file in their repo
        
    Saves intermediate data in .csv"""

    # Initializes the KEEP model and tokenizer for text/image matching.
    login(os.environ['HF_TOKEN'])  # Hugging Face token from the environment

    titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    #Conch expects 1024 patches
    conch, eval_transform = titan.return_conch()
    # Send model to GPU
    conch = conch.to("cuda")
    print("Conch sent to CUDA successfully")

    counter = 0 
    all_features = []
    for i in patch_gen:
        counter = counter + 1
        if counter % 100 == 0:
            print(f"Processed {counter} patches")
        patch = i
        img = Image.fromarray(cv2.resize(patch,(224*factor_use,224*factor_use), interpolation = cv2.INTER_AREA))

        patch_tensor = img
        patch = eval_transform(patch_tensor).unsqueeze(0).to("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            feature_emb = conch(patch)
        feature_emb = feature_emb.cpu().detach().numpy()
        all_features.append(feature_emb)
        
    all_features = np.vstack(all_features)


    prompts_all = ["An H&E image of " + i for i in disease_classification]  # Builds prompts from extracted diseases.
    if other_prompts:
        prompts_all += [f"An H&E image of {i}" for i in other_prompts]
    prompts_all.append("An H&E image of normal tissue")  # Adds a baseline prompt for normal tissue.
    for i in patch_gen:
        temp = i
        break
    patch_size_lv0 =  temp.shape[0]
    with torch.autocast('cuda', torch.float16), torch.inference_mode():
        slide_embedding = titan.encode_slide_from_patch_features(torch.from_numpy(all_features), torch.from_numpy(patch_gen.locations_df[['x','y']].values), patch_size_lv0)
    all_features = pd.DataFrame(all_features)
    slide_embedding = pd.DataFrame(slide_embedding.detach().cpu().numpy())
    all_features.to_csv(f'{output_path}/{patient_name}_TITAN_CONCH_img_features.csv')
    slide_embedding.to_csv(f'{output_path}/{patient_name}_TITAN_slide_features.csv')
    #prompts_all = np.array(prompts_all)
    #with torch.autocast('cuda', torch.float16), torch.inference_mode():
    #    classifier = titan.zero_shot_classifier(prompts_all, TEMPLATES, device='cuda')  # will take approx 3 mins
    return(all_features,slide_embedding)


