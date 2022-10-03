import sys
# sys.path
# sys.path.append('/home/jarvis/anaconda3/envs/ALBEF/lib/python3.8/site-packages/')

checkpoint_path = '../ALBEF_Download/flickr30k.pth'
config_path = 'configs/Retrieval_flickr.yaml'
IMG_DIR = '/home/nmduy/ITR2022/dataset/flickr30k_images/flickr30k-images' # Image DIR
SAVE_DIR = '/mnt/data/itr2022/ALBEF_AUG_Output/flickr30k/images' # Saving Feature Dir
JSON_TRAIN = '/home/nmduy/ITR2022/ITR_Code/JSON/train_matching_image_preprocessed_cap.json'
AUGMENT = True

device = 'cuda'
import json
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
from torch.utils.data import DataLoader

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from dataset.randaugment import RandomAugment
from PIL import Image

seed = 1509
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config['batch_size_train'] = 4
config['batch_size_test'] = 4

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) 
train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  

class re_eval_image_dataset(Dataset):
    def __init__(self, list_path, transform):        
        self.transform = transform
        self.list_path = list_path
          
    def __len__(self):
        return len(self.list_path)
    
    def __getitem__(self, index):    
        image_path = self.list_path[index]
        image_name = image_path.split('/')[-1]
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  
        return image, image_name
    

def create_dataset(list_path, transform):
    dataset = re_eval_image_dataset(list_path, transform)
    return dataset

with open(JSON_TRAIN) as f:
    json_train = json.load(f)
list_images_id = list(json_train.keys())
list_images = [f"{IMG_DIR}/{x}.jpg" for x in list_images_id]
list_images.sort()

print("Creating retrieval dataset")

if AUGMENT:
    dataset = create_dataset(list_path=list_images, transform=train_transform) # list_path is a list to images
else:
    dataset = create_dataset(list_path=list_images, transform=test_transform) # list_path is a list to images

samplers = [None]
# list of loader (can concat [train, val, test])
data_loader = create_loader([dataset],samplers,
                           batch_size=[config['batch_size_test']],
                           num_workers=[4],
                           is_trains=[False], 
                           collate_fns=[None])  
data_loader = data_loader[0]

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

model.eval(); 
for num_run in range(3):
    with torch.no_grad():
        for images, image_ids in tqdm(data_loader):
            image = images.to(device) 
            image_feat = model.visual_encoder(image)        
            image_embed = model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1) 
            image_feat = image_feat.cpu().detach().numpy()
            image_embed = image_embed.cpu().detach().numpy()
            for idx, img_id in enumerate(image_ids):
                np.savez(f'{SAVE_DIR}/{img_id[:-4]}_{num_run}.npz', feat=image_feat[idx], proj_embed=image_embed[idx])