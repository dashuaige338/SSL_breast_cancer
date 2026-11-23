import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam_survival import CLAM_SB, CLAM_MB, CLAM_SB_Surv
from utils.utils_survival import *
from utils.core_utils_survival import Accuracy_Logger
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}

    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb', 'clam_sb_surv']:
        model_dict.update({"size_arg": args.model_size})

    if args.model_type == 'clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type == 'clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'clam_sb_surv':
        model = CLAM_SB_Surv(**model_dict)
    else:  # 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)
    ckpt = torch.load(ckpt_path)
    ckpt_clean = {k.replace('.module', ''): v for k, v in ckpt.items() if 'instance_loss_fn' not in k}
    model.load_state_dict(ckpt_clean, strict=True)
    model = model.to(device).eval()
    return model


def eval_survival(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    #cindex, df = summary(model, loader, args)
    cindex, df, embeddings, attention, attention_raw = summary(model, loader, args)
    print('C-Index: {:.4f}'.format(cindex))
    return model, cindex, df, embeddings, attention, attention_raw

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def summary(model, loader, args):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slide_ids = loader.dataset.slide_data['slide_id']
    WSI_logits, slide_time, slide_event = [], [], []
    #Add the embeddings and the attention score
    WSI_embeddings = []
    WSI_attention = []
    WSI_attention_raw = []

    for batch_idx, batch in enumerate(loader):
        data, survival = batch
        y_time, y_event = survival[0].tolist()
        data = data.to(device)

        with torch.no_grad():
            #logits, _, _, _, _ = model(data)
            logits, _, _, A_raw, results_dict = model(data,return_features=True)
            WSI_logits.append(logits[0][0].item())
            slide_time.append(y_time)
            slide_event.append(y_event)
            WSI_embeddings.append(
            results_dict["features"].detach().cpu().numpy()
            )

            WSI_attention.append(
            results_dict["attention"].detach().cpu().numpy()
            )

            WSI_attention_raw.append(
            results_dict["attention_raw"].detach().cpu().numpy()
            )

    WSI_logits = np.array(WSI_logits)
    slide_time = np.array(slide_time)
    slide_event = np.array(slide_event)
    WSI_embeddings = np.concatenate(WSI_embeddings, axis=0)
    WSI_attention = [x.squeeze() for x in WSI_attention]
    WSI_attention_raw = [x.squeeze() for x in WSI_attention_raw]


    cindex = concordance_index(slide_time, -WSI_logits, slide_event)

    df = pd.DataFrame({
        'slide_id': slide_ids,
        'risk_score': WSI_logits,
        'time': slide_time,
        'event': slide_event
    })

    return cindex, df, WSI_embeddings, WSI_attention, WSI_attention_raw

