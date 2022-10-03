import os
# os.chdir('../')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
# sys.path
# sys.path.append('/home/jarvis/anaconda3/envs/DOT/lib/python3.6/site-packages/')

import argparse
import torch
import json
import glob
import pickle
import random
import time
import collections
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import ChainMap

# from horovod import torch as hvd

from uniter_model.data import ImageLmdbGroup
from transformers.tokenization_bert import BertTokenizer

from dvl.options import default_params, add_itm_params, add_logging_params, parse_with_config
from dvl.data.itm import itm_fast_collate
from dvl.models.bi_encoder import BiEncoder, setup_for_distributed_mode, load_biencoder_checkpoint
from dvl.utils import print_args, num_of_parameters, is_main_process, get_model_encoded_vecs, retrieve_query, display_img
from dvl.trainer import build_dataloader, load_dataset
from dvl.indexer.faiss_indexers import DenseFlatIndexer

from GLOBAL_VARIABLES import PROJECT_FOLDER

SAVE_DIR = '/home/nmduy/ITR2022/DOT_Output/flickr30k/texts'
MATCH_DIR = '/home/nmduy/ITR2022/dataset/flickr30k_images'
with open(f"{MATCH_DIR}/matching_image_preprocessed_cap.json") as f:
    match_data = json.load(f)
    
def train_parser(parser):
    default_params(parser)
    add_itm_params(parser)
    add_logging_params(parser)
    parser.add_argument('--teacher_checkpoint', default=None, type=str, help="")
    return parser


data_name, Full, EVAL = 'flickr', False, False

cmd = '--config /home/nmduy/ITR2022/LightningDOT/config/flickr30k_eval_config_local.json '\
      '--biencoder_checkpoint  /home/nmduy/ITR2022/DOT_Download/flickr-ft.pt ' \
      '--teacher_checkpoint /home/nmduy/ITR2022/DOT_Download/uniter-base.pt ' \
      '--img_meta /home/nmduy/ITR2022/DOT_Download/flickr_meta.json '
txt_db, img_db = '/home/nmduy/ITR2022/DOT_Download/itm_flickr30k_test.db', '/home/nmduy/ITR2022/DOT_Download/flickr30k'

parser = argparse.ArgumentParser()
parser = train_parser(parser)
args = parse_with_config(parser, cmd.split())

# options safe guard
if args.conf_th == -1:
    assert args.max_bb + args.max_txt_len + 2 <= 512
else:
    assert args.num_bb + args.max_txt_len + 2 <= 512

args.local_rank = 0
args.n_gpu = 1
args.device = torch.device("cuda", 0)

args.vector_size = 768
args.tokenizer = BertTokenizer.from_pretrained(args.txt_model_config)
print_args(args)

with open(args.itm_global_file) as f:
    args.img_meta = json.load(f)

EMBEDDED_FILE = os.path.join(os.path.dirname(args.biencoder_checkpoint), data_name + '.' + ('full' if Full else 'test') + '.pkl')

# Init Model
bi_encoder = BiEncoder(args, args.fix_img_encoder, args.fix_txt_encoder, project_dim=args.project_dim)
load_biencoder_checkpoint(bi_encoder, args.biencoder_checkpoint)

txt_model = bi_encoder.txt_model
txt_model.to(args.device)

txt_model, _ = setup_for_distributed_mode(txt_model, None, args.device, args.n_gpu, -1, args.fp16, args.fp16_opt_level)
txt_model.eval();

list_image_id = list(match_data.keys())
for image_id in tqdm(list_image_id):
    list_query = match_data[image_id]
    for idx, query in enumerate(list_query):
        input_ids = args.tokenizer.encode(query)
        input_ids = torch.LongTensor(input_ids).to(args.device).unsqueeze(0)
        attn_mask = torch.ones(len(input_ids[0]), dtype=torch.long, device=args.device).unsqueeze(0)
        pos_ids = torch.arange(len(input_ids[0]), dtype=torch.long, device=args.device).unsqueeze(0)
        # SHOULD AS FOLLOWING
        text_feat, text_embed, _ = txt_model(input_ids, attn_mask, pos_ids, None)
        
        text_feat = text_feat[0].cpu().detach().numpy()
        text_embed = text_embed[0].cpu().detach().numpy()
        attn_mask = attn_mask[0].cpu().detach().numpy()

        if idx == 0:
            concat_text_attn = [attn_mask]
            concat_text_feat = [text_feat]
            concat_text_embed = [text_embed]
        else:
            concat_text_feat.append(text_feat)
            concat_text_embed = np.concatenate((concat_text_embed, [text_embed]))
            concat_text_attn.append(attn_mask)
            
    save_name = SAVE_DIR + "/" + image_id + ".npz"
    np.savez(save_name, 
             feat=concat_text_feat, proj_embed=concat_text_embed, att=concat_text_attn)