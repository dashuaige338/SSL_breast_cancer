#Bash notes:  mkdir /gpfs/scratch/nk4167/.tiatoolbox
#Make sure you have export TIATOOLBOX_HOME=/gpfs/scratch/nk4167/.tiatoolbox
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


def run_keep(disease_classification, patch_gen, patient_name, output_path, other_prompts = None):
    """
    Runs the KEEP model to match text prompts with image patches.
    
    Args:
        disease_classification (list): List of disease classifications
        patch_gen (list of SlidingWindowPatchExtractor): Patch generator
        
    Returns:
        list: Best matching text prompts for each image patch
        
    Saves intermediate data in .csv"""

    # Initializes the KEEP model and tokenizer for text/image matching.
    model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)  # Loads the KEEP model for text/image.
    tokenizer = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)  # Tokenizes text prompts.
    model.eval()
    transform = transforms.Compose([
        # Transforms to resize and normalize image patches before feeding to the model.
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # Send model to GPU
    model = model.to("cuda")
    print("Model sent to CUDA successfully")

    prompts_all = ["An H&E image of " + i for i in disease_classification]  # Builds prompts from extracted diseases.
    if other_prompts:
        prompts_all += [f"An H&E image of {i}" for i in other_prompts]
    prompts_all.append("An H&E image of normal tissue")  # Adds a baseline prompt for normal tissue.
    #print(prompts_all)
    #Let's just do the first 50 for now
    all_matchings = []
    all_text_features = []
    all_img_features = []
    # Processes patch data for the first 50 patients to create embeddings.
    #Parse TCGA reports
    #Now iterate over the patches
    token_input = tokenizer(prompts_all,max_length=256,padding='max_length',truncation=True, return_tensors='pt').to("cuda")
    text_feature = model.encode_text(token_input)
    #print(text_feature.shape)
    counter = 0 
    for i in patch_gen:
        counter = counter + 1
        if counter % 100 == 0:
            print(f"Processed {counter} patches")
        patch = i
        img = Image.fromarray(cv2.resize(patch,(224,224), interpolation = cv2.INTER_AREA))

        img_input = transform(img).unsqueeze(0).to("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            img_feature = model.encode_image(img_input).detach()  # Detach from computation graph
        all_img_features.append(img_feature)
        
        matching = img_feature @ text_feature.T.detach()  # Detach from computation graph
        best_prompt = np.argmax(matching.cpu().numpy())
        all_matchings.append(prompts_all[best_prompt])  # Records the best matching prompt per patch
#Stack all image features
    all_img_features = torch.cat(all_img_features)
    #Stack all text features
    all_text_features = text_feature
    #Compute similarity matrix
    similarity_matrix = all_img_features@all_text_features.T
    #Detach and save embedding image features in csv file, text features and prompts with patient name
    all_img_features = all_img_features.detach().cpu().numpy()
    all_text_features = all_text_features.detach().cpu().numpy()
    all_text_features = pd.DataFrame(all_text_features)
    all_img_features = pd.DataFrame(all_img_features)
    all_img_features['x'] = patch_gen.locations_df['x']
    all_img_features['y'] = patch_gen.locations_df['y']
    all_text_features['patient'] = patient_name
    all_img_features['patient'] = patient_name
    all_text_features.to_csv(f'{output_path}/{patient_name}_KEEP_text_features.csv')
    all_img_features.to_csv(f'{output_path}/{patient_name}_KEEP_img_features.csv')
    #Return
    return(all_matchings)