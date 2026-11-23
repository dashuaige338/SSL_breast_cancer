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

#rcParam['TIATOOLBOX_HOME'] = Path('/gpfs/scratch/nk4167/.tiatoolbox')


def run_hoptimus_v0(patch_gen, patient_name,output_path):
    """
    Runs the H-Optimus model to extract image features from patches.

    Args:
        patch_gen (SlidingWindowPatchExtractor): Patch generator

    
    """
    #login(os.environ['HUGGINGFACE_TOKEN'])
    login(os.environ['HF_TOKEN'])  # Hugging Face token from the environment

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False
    )
    model.to("cuda")
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617), 
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])
    all_features = []
    counter = 0 
    for i in patch_gen:
        counter = counter + 1
        if counter % 100 == 0:
            print(f"Processed {counter} patches")
        patch = i

        img = Image.fromarray(cv2.resize(patch,(224,224), interpolation = cv2.INTER_AREA))

        img = transform(img).unsqueeze(0).to("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            feature_emb = model(img) # Extracted features (torch.Tensor) with shape [1,1024]
        feature_emb = feature_emb.cpu().detach().numpy()
        all_features.append(feature_emb)
    #Save data
    all_features = np.concatenate(all_features)
    all_features = pd.DataFrame(all_features)
    #Save including patient name
    all_features['patient'] = patient_name
    #Save CSV with patient name in filename
    all_features.to_csv(f'{output_path}/{patient_name}_hoptimusv0_features.csv')
    return(all_features)