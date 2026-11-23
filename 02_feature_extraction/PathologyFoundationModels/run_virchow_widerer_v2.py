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


def run_virchow_v2(patch_gen, patient_name, output_path):

    #login(os.environ['HUGGINGFACE_TOKEN'])
    login(os.environ['HF_TOKEN'])  # Hugging Face token from the environment

    model_virchow = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_virchow = model_virchow.eval()
    model_virchow = model_virchow.to("cuda")

    all_features = []
    counter = 0 

    transforms = create_transform(**resolve_data_config(model_virchow.pretrained_cfg, model=model_virchow))
    for i in patch_gen:
        counter = counter + 1
        if counter % 100 == 0:
            print(f"Processed {counter} patches")
        patch = i

        img = Image.fromarray(cv2.resize(patch,(224,224), interpolation = cv2.INTER_AREA), 'RGB')
        patch_model = transforms(img).unsqueeze(0).to("cuda")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model_virchow(patch_model) # Extracted features (torch.Tensor) with shape [1,1024]
        feature_emb = output.cpu().detach().numpy()
        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        # concatenate class token and average pool of patch tokens
        feature_emb = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        feature_emb = feature_emb.cpu().detach().numpy()
        all_features.append(feature_emb.squeeze())
    #Save data
    all_features = pd.DataFrame(all_features)
    #Save including patient name
    all_features['patient'] = patient_name
    #Save CSV with patient name in filename
    all_features.to_csv(f'{output_path}/{patient_name}_virchowv2_features.csv')
    return(all_features)