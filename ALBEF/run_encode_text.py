import sys
# sys.path
# sys.path.append('/home/jarvis/anaconda3/envs/ALBEF/lib/python3.8/site-packages/')

checkpoint_path = '../ALBEF_Download/flickr30k.pth'
config_path = 'configs/Retrieval_flickr.yaml'
IMG_DIR = '/home/nmduy/ITR2022/dataset/flickr30k_images/flickr30k-images'
SAVE_DIR = '/home/nmduy/ITR2022/ALBEF_Output/flickr30k/texts'

device = 'cuda'
import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

seed = 1509
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config['batch_size_train'] = 8
config['batch_size_test'] = 8

from dataset.utils import pre_caption
class re_text_dataset(Dataset):
    def __init__(self, ann_file, max_words=30):        
        is_train = 'train' in ann_file
        self.ann = json.load(open(ann_file,'r'))
        self.max_words = max_words
        self.text = []
        self.img_id = []
        n = 0
        for ann in self.ann:
            img_id = ann['image'].split('/')[-1]
            captions = ann['caption']
            if is_train:
                n += 1
                self.text.append(pre_caption(captions, self.max_words))
                self.img_id.append(img_id)
            else:
                for caption in captions:
                    self.text.append(pre_caption(caption, self.max_words))
                    self.img_id.append(img_id)
                    n += 1
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):    
        caption = self.text[index]
        img_id = self.img_id[index]
        return caption, img_id
    
print("Creating model")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = ALBEF(config=config, text_encoder='bert-base-uncased', tokenizer=tokenizer)

checkpoint = torch.load(checkpoint_path, map_location='cpu') 
state_dict = checkpoint

# reshape positional embedding to accomodate for image resolution change
pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

for key in list(state_dict.keys()):
    if 'bert' in key:
        encoder_key = key.replace('bert.','')         
        state_dict[encoder_key] = state_dict[key] 
        del state_dict[key]                
msg = model.load_state_dict(state_dict,strict=False)  

print(f'load checkpoint from {checkpoint_path}')

model = model.to(device)

train_dataset = re_text_dataset(config['train_file'][0]) 
val_dataset = re_text_dataset(config['val_file']) 
test_dataset = re_text_dataset(config['test_file'])

samplers = [None, None, None]
# list of loader (can concat [train, val, test])
data_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                           batch_size=[config['batch_size_test']]*3,
                           num_workers=[4,4,4],
                           is_trains=[False,False,False], 
                           collate_fns=[None,None,None])  

model.eval();
with torch.no_grad():
    for dtld in data_loader:
        for text, img_ids in tqdm(dtld):
            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
            text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            text_feat = text_output.last_hidden_state
            text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
            text_atts = text_input.attention_mask
            text_feat = text_feat.cpu().detach().numpy()
            text_embed = text_embed.cpu().detach().numpy()
            text_atts = text_atts.cpu().detach().numpy()

            for idx, img_id in enumerate(img_ids):
                feat = text_feat[idx] 
                proj_embed = text_embed[idx]
                att = text_atts[idx]
                caption = text[idx]
                save_name = f'{SAVE_DIR}/{img_id[:-4]}.npz'
                file_exists = os.path.exists(save_name)
                if file_exists:
                    temp = np.load(save_name)
                    temp_caption = temp['caption']
                    temp_feat = temp['feat']
                    temp_proj_embed = temp['proj_embed']
                    temp_att = temp['att']
                    if temp_caption.size == 1:
                        temp_feat = np.concatenate([[temp_feat],[feat]],axis=0)
                        temp_proj_embed = np.concatenate([[temp_proj_embed],[proj_embed]],axis=0)
                        temp_att = np.concatenate([[temp_att],[att]],axis=0)
                        temp_caption = [str(temp_caption)] + [caption]
                    else:
                        temp_feat = np.concatenate([temp_feat, [feat]], axis=0)
                        temp_proj_embed = np.concatenate([temp_proj_embed, [proj_embed]], axis=0)
                        temp_att = np.concatenate([temp_att, [att]], axis=0)
                        temp_caption = list(temp_caption) + [caption]
                else:
                    temp_feat = feat
                    temp_proj_embed = proj_embed
                    temp_att = att
                    temp_caption = caption

                np.savez(save_name, 
                         feat=temp_feat, proj_embed=temp_proj_embed,
                         att=temp_att, caption=temp_caption)