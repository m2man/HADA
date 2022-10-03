import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch
import itertools
import LiFu_m.Utils as ut
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import joblib
import torch.nn.functional as F

PERFORM_NORM = True

CONFIG_PATH = 'LiFu_m/Config/C5.yml'
config = ut.load_config(CONFIG_PATH)

JSON_TRAIN = 'JSON/train_matching_image_preprocessed_cap.json'

with open(JSON_TRAIN) as f:
    json_train = json.load(f)

list_image_id_train = list(json_train.keys())
num_img_id_train = len(list_image_id_train)

list_train = []
for x in list_image_id_train:
    cap_data = json_train[x]
    ncap = len(cap_data)
    for y in range(ncap):
        fic = f"{x}_{y}"
        list_train.append(fic)

# FIND SUM
sumx = 0
count = 0

# CAP
dataset = ut.Caption_Dataset(list_img_id_cap_id=list_train, config=config)
dataloader = ut.make_dataloader(dataset, branch='cap', batch_size=64, shuffle=False)

for idx, batch in tqdm(enumerate(dataloader)):
    cap_dict, list_img = batch
    a_t_ft = cap_dict['node_albef']
    d_t_ft = cap_dict['node_dot']
    a_t_cls = cap_dict['cls_albef']
    d_t_cls = cap_dict['cls_dot']
    if PERFORM_NORM:
        a_t_ft = F.normalize(a_t_ft, dim=1)
        d_t_ft = F.normalize(d_t_ft, dim=1)
        a_t_cls = F.normalize(a_t_cls, dim=1)
        d_t_cls = F.normalize(d_t_cls, dim=1)
    
    sumx += torch.sum(a_t_ft) + torch.sum(d_t_ft) + torch.sum(a_t_cls) + torch.sum(d_t_cls)
    count += torch.numel(a_t_ft) + torch.numel(d_t_ft) + torch.numel(d_t_cls) + torch.numel(d_t_cls)

# IMG
dataset = ut.Image_Dataset(list_img_id=list_image_id_train, config=config)
dataloader = ut.make_dataloader(dataset, branch='image', batch_size=64, shuffle=True)

for idx, batch in tqdm(enumerate(dataloader)):
    img_dict, list_id = batch
    a_i_ft = img_dict['node_albef']
    d_i_ft = img_dict['node_dot']
    a_i_cls = img_dict['cls_albef']
    d_i_cls = img_dict['cls_dot']
    
    if PERFORM_NORM:
        a_i_ft = F.normalize(a_i_ft, dim=1)
        d_i_ft = F.normalize(d_i_ft, dim=1)
        a_i_cls = F.normalize(a_i_cls, dim=1)
        d_i_cls = F.normalize(d_i_cls, dim=1)
    
    sumx += torch.sum(a_i_ft) + torch.sum(d_i_ft) + torch.sum(a_i_cls) + torch.sum(d_i_cls)
    count += torch.numel(a_i_ft) + torch.numel(d_i_ft) + torch.numel(d_i_cls) + torch.numel(d_i_cls)

del dataset, dataloader

meanx = sumx / count

# FIND SUM^2
sumx2 = 0

# CAP
dataset = ut.Caption_Dataset(list_img_id_cap_id=list_train, config=config)
dataloader = ut.make_dataloader(dataset, branch='cap', batch_size=64, shuffle=False)

for idx, batch in tqdm(enumerate(dataloader)):
    cap_dict, list_img = batch
    a_t_ft = cap_dict['node_albef']
    d_t_ft = cap_dict['node_dot']
    a_t_cls = cap_dict['cls_albef']
    d_t_cls = cap_dict['cls_dot']
    
    if PERFORM_NORM:
        a_t_ft = F.normalize(a_t_ft, dim=1)
        d_t_ft = F.normalize(d_t_ft, dim=1)
        a_t_cls = F.normalize(a_t_cls, dim=1)
        d_t_cls = F.normalize(d_t_cls, dim=1)
         
    sumx2 += torch.sum((a_t_ft-meanx)**2) + torch.sum((d_t_ft-meanx)**2) + torch.sum((a_t_cls-meanx)**2) + torch.sum((d_t_cls-meanx)**2)

# IMG
dataset = ut.Image_Dataset(list_img_id=list_image_id_train, config=config)
dataloader = ut.make_dataloader(dataset, branch='image', batch_size=64, shuffle=True)

for idx, batch in tqdm(enumerate(dataloader)):
    img_dict, list_id = batch
    a_i_ft = img_dict['node_albef']
    d_i_ft = img_dict['node_dot']
    a_i_cls = img_dict['cls_albef']
    d_i_cls = img_dict['cls_dot']

    if PERFORM_NORM:
        a_i_ft = F.normalize(a_i_ft, dim=1)
        d_i_ft = F.normalize(d_i_ft, dim=1)
        a_i_cls = F.normalize(a_i_cls, dim=1)
        d_i_cls = F.normalize(d_i_cls, dim=1)
        
    sumx2 += torch.sum((a_i_ft-meanx)**2) + torch.sum((d_i_ft-meanx)**2) + torch.sum((a_i_cls-meanx)**2) + torch.sum((d_i_cls-meanx)**2)

del dataset, dataloader
stdx = torch.sqrt(sumx2/count)

result = {}
result['a'] = {'std': stdx, 'mean': meanx}
result['d'] = {'std': stdx, 'mean': meanx}
result['all'] = {'std': stdx, 'mean': meanx}
if PERFORM_NORM:
    joblib.dump(result, 'JOBLIB/standardize_all_exact_norm.joblib')
else:
    joblib.dump(result, 'JOBLIB/standardize_all_exact.joblib')