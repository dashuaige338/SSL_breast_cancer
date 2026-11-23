import os
import torch
from huggingface_hub import login
from transformers import AutoModel 
import tifffile
from torchvision import transforms #type:ignore
import h5py
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
sys.path.append('/gpfs/data/tsirigoslab/home/jz5490/LUMC_breast_cancer/SSL/PathologyFoundationModels/prov-gigapath')
import gigapath
import gigapath.slide_encoder


def run_prov_gigapath(patch_gen,patient_name,output_path,factor_use=1):
    login(os.environ['HF_TOKEN'])  # Hugging Face token from the environment
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    tile_encoder = tile_encoder.to("cuda")
    print("tile_encoder sent to CUDA successfully")

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tile_encoder.eval()
    counter = 0 
    all_features = []
    for i in patch_gen:
        counter = counter + 1
        if counter % 100 == 0:
            print(f"Processed {counter} patches")
        patch = i
        img = Image.fromarray(cv2.resize(patch,(224*factor_use,224*factor_use), interpolation = cv2.INTER_AREA))

        patch_tensor = img
        patch = transform(patch_tensor).unsqueeze(0).to("cuda")
        with torch.no_grad():
            feature_emb = tile_encoder(patch).squeeze()
        feature_emb = feature_emb.cpu().detach().numpy()
        all_features.append(feature_emb)
        
    all_features = np.vstack(all_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slide_encoder = gigapath.slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
    slide_encoder = slide_encoder.to(device)
    slide_encoder.half()
    slide_encoder.eval()
    with torch.no_grad():
        all_features_tensor = torch.tensor(all_features, dtype=torch.float16, device=device)
        locations_tensor = torch.tensor(patch_gen.locations_df[['x', 'y']].values, dtype=torch.int64, device=device)
        slide_repr = slide_encoder(all_features_tensor.unsqueeze(0), locations_tensor.unsqueeze(0))
    if isinstance(slide_repr, list):
        slide_repr = slide_repr[0]
    all_features = pd.DataFrame(all_features)
    slide_embedding = pd.DataFrame(slide_repr.detach().cpu().numpy())
    all_features.to_csv(f'{output_path}/{patient_name}_PROVGIGAPATH_img_features.csv')
    slide_embedding.to_csv(f'{output_path}/{patient_name}_PROVGIGAPATH_slide_features.csv')
    return(all_features, slide_repr)