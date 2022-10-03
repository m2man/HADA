import __init__

import numpy as np
import os
import json
import joblib
import argparse
import numpy as np
import torch
import LiFu_m.Utils as lifum_ut
from pathlib import Path
from multiprocess import Process, Manager
import multiprocessing as mp
import time
import torch.nn.functional as F

def collect_result(result):
    global results
    results.append(result)
    
JSON_TRAIN = 'JSON/train_matching_image_preprocessed_cap.json'
JSON_VAL = 'JSON/val_matching_image_preprocessed_cap.json'
JSON_TEST = 'JSON/test_matching_image_preprocessed_cap.json'
PARA_FILE = 'JOBLIB/standardize_all_exact_norm.joblib'
try:
    para_standardize = joblib.load(PARA_FILE)
    if 'each' not in PARA_FILE:
        para_standardize['a'] = para_standardize['all']
        para_standardize['d'] = para_standardize['all']
    print("Found Standardize Para")
except:
    print("NOT Found Standardize Para")
    para_standardize = None
    
with open(JSON_TRAIN) as f:
    json_train = json.load(f)
with open(JSON_TEST) as f:
    json_test = json.load(f)
with open(JSON_VAL) as f:
    json_val = json.load(f)
    
list_image_id_train = list(json_train.keys())
list_image_id_val = list(json_val.keys())
list_image_id_test = list(json_test.keys())
num_img_id_train = len(list_image_id_train)
num_img_id_test = len(list_image_id_test)
num_img_id_val = len(list_image_id_val)

# try:
#     print("Load list_train ...")
#     list_train = joblib.load('JOBLIB/list_train.joblib')
# except:
#     print("list_train NOT FOUND --> Creating ...")
#     list_train = []
#     for x in list_image_id_train:
#         cap_data = json_train[x]
#         ncap = len(cap_data)
#         for y in range(ncap):
#             fic = f"{x}_{y}"
#             list_train.append(fic)     
#     joblib.dump(list_train, 'JOBLIB/list_train.joblib')

try:
    print("Load list_train_aug ...")
    list_train_aug = joblib.load('JOBLIB/list_train_aug.joblib')
except:
    print("list_train_aug NOT FOUND --> Creating ...")
    list_train_aug = []
    for x in list_image_id_train:
        cap_data = json_train[x]
        ncap = len(cap_data)
        for y in range(ncap):
            fic = f"{x}_{y}"
            list_train_aug.append(fic)
            for idx_aug in range(3):
                fic = f"{x}_{idx_aug}_{y}"
                list_train_aug.append(fic)
    joblib.dump(list_train_aug, 'JOBLIB/list_train_aug.joblib')

def extract_feat_index(config, idx, img_id_cap_id):
    albef_dir = config['albef_dir']
    dot_dir = config['dot_dir']
    albef_augment_dir = config['albef_augment_dir']
    wffx = config['wffx']
    wfc = config['wfc']
    wfcx = config['wfcx']
    wcc = config['wcc']
    self_loop = True
    directed_graph = config['directed_graph']
    
    img_id_cap_id_split = img_id_cap_id.split('_')
    if len(img_id_cap_id_split) == 3: # include augment idx
        img_id, aug_id, cap_id = img_id_cap_id_split
    else: # normal  
        img_id, cap_id = img_id_cap_id_split
        aug_id = None
    cap_id = int(cap_id)
    # ALBEF
    if aug_id is not None:
        albef_img = np.load(f"{albef_augment_dir}/images/{img_id}_{aug_id}.npz")
    else:
        albef_img = np.load(f"{albef_dir}/images/{img_id}.npz")
    albef_cap = np.load(f"{albef_dir}/texts/{img_id}.npz")
    # albef_img_feat = albef_img['feat'][1:] # the CLS is projected with additional layer (576, 768)
    albef_img_feat = albef_img['feat'][0:] # the CLS is projected with additional layer (1_cls+576, 768)
    albef_img_cls = albef_img['proj_embed'] # the projected CLS (256,)
    albef_cap_feat = albef_cap['feat'][cap_id] # 30, 768
    albef_cap_cls = albef_cap['proj_embed'][cap_id] # (256,)
    albef_cap_att = albef_cap['att'][cap_id] # (30,) binary vector indicate len of a caption 1, 1, 1, 0, 0...
    len_cap = sum(albef_cap_att) # include the CLS
    # albef_cap_feat = albef_cap_feat[1:len_cap] # the CLS is projected with additional layer (len_cap-1, 768)
    albef_cap_feat = albef_cap_feat[0:len_cap] # --> 1st element is CLS (before projected as cap_cls) (len_cap, 768)
    # DOT
    dot_img = np.load(f"{dot_dir}/images/{img_id}.npz")
    dot_cap = np.load(f"{dot_dir}/texts/{img_id}.npz",allow_pickle=True)
    # dot_img_feat = dot_img['feat'][1:] # the CLS is projected with additional layer (n_obj, 768)
    dot_img_feat = dot_img['feat'][0:] # the CLS is projected with additional layer (1_cls+n_obj, 768)
    dot_img_cls = dot_img['proj_embed'] # the projected CLS (768,)
    dot_cap_feat = dot_cap['feat'][cap_id] # include CLS (n_len, 768)
    dot_cap_att = dot_cap['att'][cap_id] # (n_len)
    len_cap = sum(dot_cap_att) # include the CLS
    # dot_cap_feat = dot_cap_feat[1:len_cap] # (len_cap-1, 768)
    dot_cap_feat = dot_cap_feat[0:len_cap] # (len_cap, 768) # --> 1st element is CLS (before projected as cap_cls)
    dot_cap_cls = dot_cap['proj_embed'][cap_id] # (768,)
    # # Convert to tensor
    # albef_cap_feat = torch.tensor(albef_cap_feat)
    # albef_cap_cls = torch.tensor(albef_cap_cls)
    # albef_img_feat = torch.tensor(albef_img_feat)
    # albef_img_cls = torch.tensor(albef_img_cls)
    # dot_cap_feat = torch.tensor(dot_cap_feat)
    # dot_cap_cls = torch.tensor(dot_cap_cls)
    # dot_img_feat = torch.tensor(dot_img_feat)
    # dot_img_cls = torch.tensor(dot_img_cls)
    if config['util_norm']:
        # albef_cap_feat = F.normalize(albef_cap_feat, dim=1)
        # albef_cap_cls = F.normalize(albef_cap_cls.view(1,-1)).squeeze()
        # albef_img_feat = F.normalize(albef_img_feat, dim=1)
        # albef_img_cls = F.normalize(albef_img_cls.view(1,-1)).squeeze()
        # dot_cap_feat = F.normalize(dot_cap_feat, dim=1)
        # dot_cap_cls = F.normalize(dot_cap_cls.view(1,-1)).squeeze()
        # dot_img_feat = F.normalize(dot_img_feat, dim=1)
        # dot_img_cls = F.normalize(dot_img_cls.view(1,-1)).squeeze()
        albef_cap_feat = albef_cap_feat/(np.linalg.norm(albef_cap_feat, axis=1).reshape(-1,1))
        albef_cap_cls = albef_cap_cls / np.linalg.norm(albef_cap_cls)
        albef_img_feat = albef_img_feat/(np.linalg.norm(albef_img_feat, axis=1).reshape(-1,1))
        albef_img_cls = albef_img_cls / np.linalg.norm(albef_img_cls)
        dot_cap_feat = dot_cap_feat/(np.linalg.norm(dot_cap_feat, axis=1).reshape(-1,1))
        dot_cap_cls = dot_cap_cls / np.linalg.norm(dot_cap_cls)
        dot_img_feat = dot_img_feat/(np.linalg.norm(dot_img_feat, axis=1).reshape(-1,1))
        dot_img_cls = dot_img_cls / np.linalg.norm(dot_img_cls)
    # Normalize data
    if para_standardize is not None:
        if 'a_i' in list(para_standardize.keys()):
            albef_cap_feat = lifum_ut.do_standardize(albef_cap_feat, para_standardize['a_t']['mean'].item(), para_standardize['a_t']['std'].item())
            albef_img_feat = lifum_ut.do_standardize(albef_img_feat, para_standardize['a_i']['mean'].item(), para_standardize['a_i']['std'].item())
            dot_cap_feat = lifum_ut.do_standardize(dot_cap_feat, para_standardize['d_t']['mean'].item(), para_standardize['d_t']['std'].item())
            dot_img_feat = lifum_ut.do_standardize(dot_img_feat, para_standardize['d_i']['mean'].item(), para_standardize['d_i']['std'].item())
        else:
            albef_cap_feat = lifum_ut.do_standardize(albef_cap_feat, para_standardize['a']['mean'].item(), para_standardize['a']['std'].item())
            albef_cap_cls = lifum_ut.do_standardize(albef_cap_cls, para_standardize['a']['mean'].item(), para_standardize['a']['std'].item())
            albef_img_feat = lifum_ut.do_standardize(albef_img_feat, para_standardize['a']['mean'].item(), para_standardize['a']['std'].item())
            albef_img_cls = lifum_ut.do_standardize(albef_img_cls, para_standardize['a']['mean'].item(), para_standardize['a']['std'].item())
            dot_cap_feat = lifum_ut.do_standardize(dot_cap_feat, para_standardize['d']['mean'].item(), para_standardize['d']['std'].item())
            dot_cap_cls = lifum_ut.do_standardize(dot_cap_cls, para_standardize['d']['mean'].item(), para_standardize['d']['std'].item())
            dot_img_feat = lifum_ut.do_standardize(dot_img_feat, para_standardize['d']['mean'].item(), para_standardize['d']['std'].item())
            dot_img_cls = lifum_ut.do_standardize(dot_img_cls, para_standardize['d']['mean'].item(), para_standardize['d']['std'].item())

    # IMAGE GRAPH
    # img_node = torch.cat((albef_img_feat, dot_img_feat))
    n_dot = dot_img_feat.shape[0]
    n_albef = albef_img_feat.shape[0]
    albef_index = [x for x in range(n_albef)]
    dot_index = [n_albef+x for x in range(n_dot)]
    # create edge here
    c1 = [albef_index[0]]
    c2 = [dot_index[0]]
    f1 = albef_index[1:]
    f2 = dot_index[1:]
    f1_c1 = lifum_ut.create_index_from_2_list(f1, c1, dual_index=not directed_graph, self_loop=self_loop) # (2, n_edge)
    f2_c2 = lifum_ut.create_index_from_2_list(f2, c2, dual_index=not directed_graph, self_loop=self_loop)
    f2_c1 = lifum_ut.create_index_from_2_list(f2, c1, dual_index=not directed_graph, self_loop=False)
    f1_c2 = lifum_ut.create_index_from_2_list(f1, c2, dual_index=not directed_graph, self_loop=False)
    c1_c2 = lifum_ut.create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
    f1_f2 = lifum_ut.create_index_from_2_list(f1, f2, dual_index=True, self_loop=False)
    edge_index = np.concatenate((f1_c1, f2_c2, c1_c2, f1_c2, f2_c1, f1_f2), axis=1)
    # create edge attr
    f1_c1_attr = [wfc for x in range(f1_c1.shape[1])]
    f2_c2_attr = [wfc for x in range(f2_c2.shape[1])]
    if self_loop:
        f1_c1_attr[-1] = wcc
        f2_c2_attr[-1] = wcc
    c1_c2_attr = [wcc for x in range(c1_c2.shape[1])]
    f1_c2_attr = [wfcx for x in range(f1_c2.shape[1])]
    f2_c1_attr = [wfcx for x in range(f2_c1.shape[1])]
    f1_f2_attr = [wffx for x in range(f1_f2.shape[1])]
    edge_attr = np.array(f1_c1_attr + f2_c2_attr + c1_c2_attr + f1_c2_attr + f2_c1_attr + f1_f2_attr)
    # drop ffx or not
    drop_idx = np.where(edge_attr==-1)[0]
    # img_edge_attr = torch.tensor(np.delete(edge_attr, drop_idx), dtype=torch.float)
    # img_edge_index = torch.tensor(np.delete(edge_index, drop_idx, 1))
    img_edge_attr = np.delete(edge_attr, drop_idx)
    img_edge_index = np.delete(edge_index, drop_idx, 1)

    # CAPTION GRAPH
    # cap_node = torch.cat((albef_cap_feat, dot_cap_feat))
    n_dot = dot_cap_feat.shape[0]
    n_albef = albef_cap_feat.shape[0]
    albef_index = [x for x in range(n_albef)]
    dot_index = [n_albef+x for x in range(n_dot)]
    # create edge here
    c1 = [albef_index[0]]
    c2 = [dot_index[0]]
    f1 = albef_index[1:]
    f2 = dot_index[1:]
    f1_c1 = lifum_ut.create_index_from_2_list(f1, c1, dual_index=not directed_graph, self_loop=self_loop) # (2, n_edge)
    f2_c2 = lifum_ut.create_index_from_2_list(f2, c2, dual_index=not directed_graph, self_loop=self_loop)
    f2_c1 = lifum_ut.create_index_from_2_list(f2, c1, dual_index=not directed_graph, self_loop=False)
    f1_c2 = lifum_ut.create_index_from_2_list(f1, c2, dual_index=not directed_graph, self_loop=False)
    c1_c2 = lifum_ut.create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
    f1_f2 = lifum_ut.create_index_from_2_list(f1, f2, dual_index=True, self_loop=False)
    edge_index = np.concatenate((f1_c1, f2_c2, c1_c2, f1_c2, f2_c1, f1_f2), axis=1)
    # create edge attr
    f1_c1_attr = [wfc for x in range(f1_c1.shape[1])]
    f2_c2_attr = [wfc for x in range(f2_c2.shape[1])]
    if self_loop:
        f1_c1_attr[-1] = wcc
        f2_c2_attr[-1] = wcc
    c1_c2_attr = [wcc for x in range(c1_c2.shape[1])]
    f1_c2_attr = [wfcx for x in range(f1_c2.shape[1])]
    f2_c1_attr = [wfcx for x in range(f2_c1.shape[1])]
    f1_f2_attr = [wffx for x in range(f1_f2.shape[1])]
    edge_attr = np.array(f1_c1_attr + f2_c2_attr + c1_c2_attr + f1_c2_attr + f2_c1_attr + f1_f2_attr)
    # drop ffx or not
    drop_idx = np.where(edge_attr==-1)[0]
    # cap_edge_attr = torch.tensor(np.delete(edge_attr, drop_idx), dtype=torch.float)
    # cap_edge_index = torch.tensor(np.delete(edge_index, drop_idx, 1))
    cap_edge_attr = np.delete(edge_attr, drop_idx)
    cap_edge_index = np.delete(edge_index, drop_idx, 1)

    # albef_img_cls = albef_img_cls.view(1,-1)
    # dot_img_cls = dot_img_cls.view(1,-1)
    # albef_cap_cls = albef_cap_cls.view(1,-1)
    # dot_cap_cls = dot_cap_cls.view(1,-1)
    
    albef_img_cls = albef_img_cls.reshape(1,-1)
    dot_img_cls = dot_img_cls.reshape(1,-1)
    albef_cap_cls = albef_cap_cls.reshape(1,-1)
    dot_cap_cls = dot_cap_cls.reshape(1,-1)
    
    dict_return = {}
    dict_return['img'] = {'node_albef': albef_img_feat, # (1+576, 768)
                          'node_dot': dot_img_feat, # (1+n_obj, 768)
                          'edge_index': img_edge_index, #(2, len_edge)
                          'edge_attr': img_edge_attr, # (len_edge)
                          'cls_albef': albef_img_cls, # (1,256) 
                          'cls_dot': dot_img_cls} # (1, 768)

    dict_return['cap'] = {'node_albef': albef_cap_feat, # (1+len_cap)
                          'node_dot': dot_cap_feat, # (1+len_cap, 768)
                          'edge_index': cap_edge_index, #(2, len_edge)
                          'edge_attr': cap_edge_attr, # (len_edge)
                          'cls_albef': albef_cap_cls, # (1,256) 
                          'cls_dot': dot_cap_cls} # (1, 768)
    dict_return['id'] = f"{img_id}_{cap_id}"
    return (idx, dict_return)
    
def run_extract_feat_dir(args):
    config_path = args.config_path
    model_type = config_path.split('/')[0] # LiFu_m, ...
    
    save_dir = f"Features/{model_type}"
    Path(f"{save_dir}").mkdir(parents=True, exist_ok=True)
    
    config = lifum_ut.load_config(config_path)
    
    if 'norm' in PARA_FILE:
        config['util_norm'] = True
    else:
        config['util_norm'] = False
    list_img_id_cap_id = list_train_aug.copy()
        
    print(f"Creating Feature Dir in Dataset ... ")
    save_name = config_path.split('/')[-1]
    save_name = f"{save_name.split('.')[0]}"
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    step = 20000
    for idx in range(0,len(list_img_id_cap_id), step):
        print(f"Processing {idx} ...")    
        pool = mp.Pool(mp.cpu_count())
        result_objects = [pool.apply_async(extract_feat_index, args=(config, i, cid)) for i, cid in enumerate(list_img_id_cap_id[idx:min(idx+step, len(list_img_id_cap_id))])]
        feat_dict = [r.get()[1] for r in result_objects]
        pool.close()
        pool.join()
        torch.save(feat_dict, f"{save_dir}/{save_name}_{idx}.pth")
    
    print(f"Created Feature Dir in Dataset !!!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='LiFu_m/Config/C5.yml', help='yml file of the config')
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    # TOPK = args.top_k
    print(f"CONFIG: {CONFIG_PATH.split('/')[-1]}")
    timestampTime = time.strftime("%H%M%S")
    print(timestampTime)
    run_extract_feat_dir(args)
    timestampTime = time.strftime("%H%M%S")
    print(timestampTime)