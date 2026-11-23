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


def run_uni_v2(patch_gen, patient_name,output_path):
    #login(os.environ['HUGGINGFACE_TOKEN'])
    login(os.environ['HF_TOKEN'])  # Hugging Face token from the environment
    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
    timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    model = model.to("cuda")

    all_features = []
    counter = 0
    for i in patch_gen:
        counter = counter + 1
        if counter % 100 == 0:
            print(f"Processed {counter} patches")
        patch = i

        img = Image.fromarray(cv2.resize(patch,(224,224), interpolation = cv2.INTER_AREA))
        patch = transform(img).unsqueeze(0).to("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            feature_emb = model(patch) # Extracted features (torch.Tensor) with shape [1,1024]
        feature_emb = feature_emb.cpu().detach().numpy()
        all_features.append(feature_emb)
    #Save data
    all_features = np.concatenate(all_features)
    all_features = pd.DataFrame(all_features)
    #Save including patient name
    all_features['patient'] = patient_name
    #Save CSV with patient name in filename
    all_features.to_csv(f'{output_path}/{patient_name}_univ2_features.csv')
    return(all_features)