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

PERFORM_NORM = False

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
sum_img_albef = 0
count_img_albef = 0
sum_cap_albef = 0
count_cap_albef = 0

sum_img_dot = 0
count_img_dot = 0
sum_cap_dot = 0
count_cap_dot = 0


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
    
    sum_cap_albef += torch.sum(a_t_ft)
    count_cap_albef += torch.numel(a_t_ft)
    sum_cap_dot += torch.sum(d_t_ft) 
    count_cap_dot += torch.numel(d_t_ft)

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
    
    sum_img_albef += torch.sum(a_t_ft)
    count_img_albef += torch.numel(a_t_ft)
    sum_img_dot += torch.sum(d_t_ft) 
    count_img_dot += torch.numel(d_t_ft)

del dataset, dataloader

mean_img_albef = sum_img_albef / count_img_albef
mean_img_dot = sum_img_dot / count_img_dot
mean_cap_albef = sum_cap_albef / count_cap_albef
mean_cap_dot = sum_cap_dot / count_cap_dot

# FIND SUM^2
sum2_img_albef = 0
sum2_img_dot = 0
sum2_cap_albef = 0
sum2_cap_dot = 0

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
    
    sum2_cap_albef += torch.sum((a_t_ft-mean_cap_albef)**2)
    sum2_cap_dot += torch.sum((d_t_ft-mean_cap_dot)**2)

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
    
    sum2_img_albef += torch.sum((a_t_ft-mean_img_albef)**2)
    sum2_img_dot += torch.sum((d_t_ft-mean_img_dot)**2)

del dataset, dataloader
std_img_albef = torch.sqrt(sum2_img_albef/count_img_albef)
std_img_dot = torch.sqrt(sum2_img_dot/count_img_dot)
std_cap_albef = torch.sqrt(sum2_cap_albef/count_cap_albef)
std_cap_dot = torch.sqrt(sum2_cap_dot/count_cap_dot)

result = {}
result['a_i'] = {'std': std_img_albef, 'mean': mean_img_albef}
result['d_i'] = {'std': std_img_dot, 'mean': mean_img_dot}
result['a_t'] = {'std': std_cap_albef, 'mean': mean_cap_albef}
result['d_t'] = {'std': std_cap_dot, 'mean': mean_cap_dot}

if PERFORM_NORM:
    joblib.dump(result, 'JOBLIB/standardize_each_norm.joblib')
else:
    joblib.dump(result, 'JOBLIB/standardize_each.joblib')