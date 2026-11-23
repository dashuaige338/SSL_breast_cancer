import os
import pynvml
from utils import *
from PathologyFoundationModels.run_virchow_widerer_v2 import *
from PathologyFoundationModels.run_uni_widerer_v15 import *
#from PathologyFoundationModels.run_hoptimusv0 import *
#from PathologyFoundationModels.run_keep import *

#Download a TCGA slide SVS using their API
def download_bs_svs(patient_id,output_dir='./output'):
    """
    Searches GDC for files matching BS slides and downloads them.
    
    Args:
        patient_id (str): TCGA patient identifier
        output_dir (str): Directory to save downloaded slides
    """
    api_url = "https://api.gdc.cancer.gov/files"
    query = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.submitter_id", "value": [patient_id]}},
                {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
                {"op": "=", "content": {"field": "file_name", "value": "*BS*"}}  # Strict BS filter
            ]
        },
        "fields": "file_id,file_name",
        "format": "JSON",
        "size": 1000
    }

    response = requests.post(api_url, json=query).json()
    files = response.get('data', {}).get('hits', [])
    
    if not files:
        raise ValueError(f"No BS slides found for {patient_id}")

    os.makedirs(output_dir, exist_ok=True)
    downloaded_files = []
    print(files)
    for f in files:
        # Builds the download URL and streams the file to disk.
        dl_url = f"https://api.gdc.cancer.gov/data/{f['file_id']}"
        path = os.path.join(output_dir, f['file_name'])
        
        with requests.get(dl_url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        downloaded_files.append(path)
    
    return downloaded_files

def generate_wsi_patches(wsi_image, wsi_mask, patch_size=448, overlap=256//8, output_dir="./output", 
                         tissue_threshold=0.75, resolution=0.51, min_patches=100):
    """
    Generate overlapping tissue patches from WSIs with quality control.
    
    Args:
        image_path (str): Path to input SVS image
        mask_path (str): Path to segmentation mask
        patch_size (int): Patch size in pixels (square)
        overlap (int): Overlap between patches in pixels
        output_dir (str): Directory to save patches
        tissue_threshold (float): Minimum tissue content (0-1)
        resolution (float): Microns per pixel (default 0.5 = 20x)
        min_patches (int): Minimum number of patches to collect
    
    Returns:
        dict: Metadata about generated patches
    """
    
    # Verify matching dimensions at target resolution
    img_dims = wsi_image.slide_dimensions(resolution=resolution, units='mpp')
    mask_dims = wsi_mask.slide_dimensions(resolution=resolution, units='mpp')
    print(img_dims)
    print(mask_dims)
    # Calculate stride and initialize patch generator
    stride = patch_size - overlap
    patch_gen = SlidingWindowPatchExtractor(
        # Configures sliding window on WSI image and mask.
        input_img=wsi_image,
        patch_size=(patch_size, patch_size),
        stride=(stride, stride),
        input_mask=wsi_mask,
        min_mask_ratio=tissue_threshold
    )
    return(patch_gen)


def download_slide(patient_id):
    """
    Downloads a TCGA slide SVS and prepares patches for processing.
    Output directory is named after the patient ID.
    
    Args:
        patient_id (str): TCGA patient identifier

    Returns:
        tuple: Path to SVS, WSIReader object, tissue mask, patch generator
    """
    # Orchestrates download of slide SVS and preparation of patches.
    svs_path = download_bs_svs(patient_id, output_dir=patient_id + 'output')[0]
    wsi = WSIReader.open(svs_path)
    mask = wsi.tissue_mask(resolution=1, units="power")
    patch_gen = generate_wsi_patches(wsi, mask, resolution= wsi.info.mpp[0])
    return(svs_path,wsi,mask,patch_gen)

def get_existing_slide(svs_path):
    #function to extract the patch features if you have exisiting slides
    wsi = WSIReader.open(svs_path)
    mask = wsi.tissue_mask(resolution=1, units="power")
    patch_gen = generate_wsi_patches(wsi, mask, resolution= wsi.info.mpp[0])
    return patch_gen


def download_tcga_pathology():
    """
    Downloads TCGA pathology reports for downstream analysis.
    
    Returns:
        pd.DataFrame: TCGA pathology reports
    """
    # Downloads TCGA pathology reports for downstream analysis.
    os.system("curl -L https://data.mendeley.com/public-files/datasets/hyg5xkznpx/files/60abe141-9352-4a54-943c-3d015eabefea/file_downloaded --output TCGA_Reports.csv.zip")
    os.system("unzip -qq -o TCGA_Reports.csv.zip")
    tcga_reports = pd.read_csv("TCGA_Reports.csv")
    return tcga_reports

def pathchat_for_poor_people(text):
    """Uses DeepSeek API to extract simplified disease phrases from pathology reports.
    
    Args:
        text (str): Pathology report text
        
    Returns:
        str: Simplified disease phrase
    """
    session = requests.Session()
    DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
    # Calls the DeepSeek API to return a simplified disease phrase.
    prompt_prefix="Given the following text, what disease is this patient suffering from, answer in a single phrase with the disease in a simpified form. Add no other words other than the disease itself. Here is the report:"
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{
            "role": "user", 
            "content": f"{prompt_prefix}\n\n{ text}"
        }]
    }
    response = session.post(
                endpoint,
                headers=headers,
                json=payload
            )
    response.raise_for_status()
    response_json = response.json()
    content = response_json['choices'][0]['message']['content']
    return(content)


def get_gpu_utilisation():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        torch_gpu_id = torch.cuda.current_device()
        pynvml.nvmlInit() 
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        nvml_gpu_id = int(devices[torch_gpu_id]) 
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info_used = info.used//1024**2 
        info_total = info.total//1024**2

        print(f"GPU {nvml_gpu_id} memory occupied: {info_used}/{info_total} MB\
        = {np.round((info_used*100)/info_total,2)}%.")

        pynvml.nvmlShutdown()



