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

IMG_DIR = '/home/nmduy/ITR2022/dataset/flickr30k_images/flickr30k-images'
FT_DIR = '/home/nmduy/ITR2022/BUA_Output/flickr30k_101_ra'
SAVE_DIR = '/home/nmduy/ITR2022/DOT_Output/flickr30k/images_101'
CONF_TH = 0.2
MIN_BB = 10
MAX_BB = 100

def train_parser(parser):
    default_params(parser)
    add_itm_params(parser)
    add_logging_params(parser)
    parser.add_argument('--teacher_checkpoint', default=None, type=str, help="")
    return parser

def process_bb(bb):
    img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
    num_bb = img_bb.size(0)
    return img_bb, num_bb

def process_sample(bbox_ft, bbox_pos):
    img_input_ids = torch.Tensor([101]).long()
    img_feat = bbox_ft
    img_pos_feat, num_bb = process_bb(bbox_pos)
    attn_masks_img = torch.ones(num_bb+1, dtype=torch.long)
    out_size = num_bb + 1
    num_bbs = [num_bb]
    # Convert to batch
    img_input_ids = img_input_ids.unsqueeze(0)
    img_feat = img_feat.unsqueeze(0)
    img_pos_feat = img_pos_feat.unsqueeze(0)
    attn_masks_img = attn_masks_img.unsqueeze(0)
    img_position_ids = torch.arange(0, img_input_ids.size(1), dtype=torch.long).unsqueeze(0)
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(len(num_bbs), 1)
    batch = {'input_ids': img_input_ids,
            'position_ids': img_position_ids,
            'attention_mask': attn_masks_img,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feat,
            'img_masks': None,
            'gather_index': gather_index
            }
    return batch

def compute_num_bb(confs, conf_th, min_bb, max_bb):
    num_bb = max(min_bb, (confs > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return num_bb

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

img_model = bi_encoder.img_model
img_model.to(args.device)

img_model, _ = setup_for_distributed_mode(img_model, None, args.device, args.n_gpu, -1, args.fp16, args.fp16_opt_level)
img_model.eval();

# txt_model, _ = setup_for_distributed_mode(txt_model, None, args.device, args.n_gpu, -1, args.fp16, args.fp16_opt_level)
# txt_model.eval();

list_image = [f for f in glob.glob(f"{IMG_DIR}/*.jpg")]
list_image = [x.split('/')[-1] for x in list_image]
list_image.sort()

for idx_image in tqdm(list_image):
    # 1. Load BBOX and FT
    # 1.1 BBOX in format of X,Y,X,Y same with this project
    idx_image_no = idx_image.split('.')[0]
    ft_info = np.load(f"{FT_DIR}/{idx_image_no}.npz", allow_pickle=True)
    # ft_info = ft['info'].tolist()
    # ft_info keys: ['image_id', 'image_h', 'image_w', 'num_bbox', 'bbox_pos', 'bbox_ft',
    #                'objects_id', 'objects_conf', 'attrs_id', 'attrs_conf']
    
    # 2. Filter CONF_TH = 0.2
    # 2.1 Sort bbox and ft regions base on conf_th
    # 2.2 Run following function to find suitable num_bb base on CONF_TH
    # # def compute_num_bb(confs, conf_th, min_bb, max_bb):
    #     num_bb = max(min_bb, (confs > conf_th).sum())
    #     num_bb = min(max_bb, num_bb)
    #     return num_bb
    # 2.3 Cut of bbox and ft regions (and corresponding attributes, scores, ...)

    objects_conf = ft_info['objects_conf']
    attrs_conf = ft_info['attrs_conf']
    objects_id = ft_info['objects_id']
    attrs_id = ft_info['attrs_id']
    bbox_pos = ft_info['bbox_pos']
    bbox_ft = ft_info['bbox_ft']
    mean_conf = (2*objects_conf + attrs_conf)/3
    
    ascending_idx = np.argsort(mean_conf) # ascending order
    descending_idx = ascending_idx[::-1]
    objects_conf = objects_conf[descending_idx]
    attrs_conf = attrs_conf[descending_idx]
    objects_id = objects_id[descending_idx]
    attrs_id = attrs_id[descending_idx]
    bbox_pos = bbox_pos[descending_idx]
    bbox_ft = bbox_ft[descending_idx]
    
    num_bb = compute_num_bb(mean_conf, CONF_TH, MIN_BB, MAX_BB)
    objects_conf = objects_conf[:num_bb]
    attrs_conf = attrs_conf[:num_bb]
    objects_id = objects_id[:num_bb]
    attrs_id = attrs_id[:num_bb]
    bbox_pos = bbox_pos[:num_bb]
    bbox_ft = bbox_ft[:num_bb]
    
    # 3. Normalize BBOX
    # 3.1 X = X / image_w  (bbox[0] and bbox[2]) , Y = Y / image_h (bbox[1] and bbox[3])
    # 3.2 Add W and H to BBOX  bbox[4] = bbox[2] - bbox[0], bbox [5] = bbox[3] - bbox[1]

    bbox_pos[:,0] = bbox_pos[:,0] / ft_info['image_w']
    bbox_pos[:,2] = bbox_pos[:,2] / ft_info['image_w']
    bbox_pos[:,1] = bbox_pos[:,1] / ft_info['image_h']
    bbox_pos[:,3] = bbox_pos[:,3] / ft_info['image_h']
    
    W = bbox_pos[:,2] - bbox_pos[:,0]
    H = bbox_pos[:,3] - bbox_pos[:,1]
    W = np.reshape(W, (-1,1))
    H = np.reshape(H,(-1,1))
    bbox_pos = np.concatenate((bbox_pos,W,H),axis=1)
    
    # 4. Run process_bb and process_process_sample
    data_batch = process_sample(torch.tensor(bbox_ft), torch.tensor(bbox_pos))
    
    # 5. Run model encode
    input_ids = data_batch['input_ids'].to(args.device)
    attention_mask = data_batch['attention_mask'].to(args.device)
    position_ids = data_batch['position_ids'].to(args.device)
    img_feat = data_batch['img_feat'].to(args.device)
    img_pos_feat = data_batch['img_pos_feat'].to(args.device)
    img_masks = data_batch['img_masks']
    gather_index = data_batch['gather_index'].to(args.device)
    with torch.no_grad():
        sequence_output, pooled_output, hidden_states = img_model(input_ids, attention_mask, position_ids,
                                                                  img_feat, img_pos_feat, img_masks,
                                                                  gather_index)
        
    image_feat = sequence_output[0].cpu().detach().numpy()
    image_embed = pooled_output[0].cpu().detach().numpy()
    
    np.savez(f'{SAVE_DIR}/{idx_image_no}.npz', feat=image_feat, proj_embed=image_embed)