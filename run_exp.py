import __init__

import json
import joblib
import argparse
import numpy as np
from HADA_m.Retrieval_Utils import i2t, t2i, evaluate_recall
from HADA_m.Utils import write_to_file
import torch
import mlflow

import Baseline_m.Utils as blm_ut
import Baseline_m_extend.Utils as blme_ut
import HADA_m.Utils as lifum_ut
import HADA_m_extend.Utils as lifume_ut

from Baseline_m.Controller import Controller as Blm_ctr
from Baseline_m_extend.Controller import Controller as Blme_ctr
from HADA_m.Controller import Controller as Lifum_ctr
from HADA_m_extend.Controller import Controller as Lifume_ctr

mlflow.set_tracking_uri('http://localhost:1409')

JSON_TRAIN = 'JSON/train_matching_image_preprocessed_cap.json'
JSON_VAL = 'JSON/val_matching_image_preprocessed_cap.json'
JSON_TEST = 'JSON/test_matching_image_preprocessed_cap.json'
# PARA_FILE = 'JOBLIB/standardize_each.joblib'
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

try:
    print("Load list_train ...")
    list_train = joblib.load('JOBLIB/list_train.joblib')
except:
    print("list_train NOT FOUND --> Creating ...")
    list_train = []
    for x in list_image_id_train:
        cap_data = json_train[x]
        ncap = len(cap_data)
        for y in range(ncap):
            fic = f"{x}_{y}"
            list_train.append(fic)     
    joblib.dump(list_train, 'JOBLIB/list_train.joblib')

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

try:
    print("Load list_val...")
    list_val = joblib.load('JOBLIB/list_val.joblib')
except:
    print("list_val NOT FOUND --> Creating ...")
    list_val = []
    for x in list_image_id_val:
        cap_data = json_val[x]
        ncap = len(cap_data)
        for y in range(ncap):
            fic = f"{x}_{y}"
            list_val.append(fic)
    joblib.dump(list_val, 'JOBLIB/list_val.joblib')

try:
    print("Load list_test...")
    list_test = joblib.load('JOBLIB/list_test.joblib')
except:
    print("list_test NOT FOUND --> Creating ...")
    list_test = []    
    for x in list_image_id_test:
        cap_data = json_test[x]
        ncap = len(cap_data)
        for y in range(ncap):
            fic = f"{x}_{y}"
            list_test.append(fic)
    joblib.dump(list_test, 'JOBLIB/list_test.joblib')
        
def run_train(args):
    print(f"RUN TRAIN")
    config_path = args.config_path
    model_type = config_path.split('/')[0]
    config_name = config_path.split('/')[-1][:-4]
    
    if model_type == 'HADA_m':
        config = lifum_ut.load_config(config_path)
    elif model_type == 'HADA_m_extend':
        config = lifume_ut.load_config(config_path)
    elif model_type == 'Baseline_m':
        config = blm_ut.load_config(config_path)
    elif model_type == 'Baseline_m_extend':
        config = blme_ut.load_config(config_path)
    
    if 'norm' in PARA_FILE:
        config['util_norm'] = True
    else:
        config['util_norm'] = False
    config['model_type'] = model_type
    config['config_path'] = config_path
    
    niters = int(int(np.ceil(len(list_train) / config['batch_size'])))
    
    if config['Tmax'] > 0:
        config['Tmax'] = config['Tmax'] * niters
        
    if model_type == 'HADA_m':
        controller = Lifum_ctr(config)
    elif model_type == 'HADA_m_extend':
        controller = Lifume_ctr(config)
    elif model_type == 'Baseline_m':
        controller = Blm_ctr(config)
    elif model_type == 'Baseline_m_extend':
        controller = Blme_ctr(config)
    
    total_para = controller.count_parameters()
    print(f"Trainable Paras: {total_para}")
    controller.train(para_standardize=para_standardize,
                     train_img_id=list_image_id_train, 
                     train_cap_id=list_train,
                     train_img_aug_cap_id=None,
                     val_img_id=list_image_id_val, 
                     val_cap_id=list_val,
                     num_epoch=config['num_epoch'], model_name=config_name)
    

def run_evaluate(args):
    config_path = args.config_path
    model_type = config_path.split('/')[0]
    config_name = config_path.split('/')[-1][:-4]
    
    print(f"PERFORM EVALUATE")
    if model_type == 'HADA_m':
        config = lifum_ut.load_config(config_path)
    elif model_type == 'HADA_m_extend':
        config = lifume_ut.load_config(config_path)
    elif model_type == 'Baseline_m':
        config = blm_ut.load_config(config_path)
    elif model_type == 'Baseline_m_extend':
        config = blme_ut.load_config(config_path)
        
    save_path = f"{model_type}/{config['out_dir']}/{config_name}/best.pth.tar"
    config['model_type'] = model_type
    config['config_path'] = config_path
    
    if 'norm' in PARA_FILE:
        config['util_norm'] = True
    else:
        config['util_norm'] = False
        
    if model_type == 'HADA_m':
        controller = Lifum_ctr(config)
    elif model_type == 'HADA_m_extend':
        controller = Lifume_ctr(config)
    elif model_type == 'Baseline_m':
        controller = Blm_ctr(config)
    elif model_type == 'Baseline_m_extend':
        controller = Blme_ctr(config)
        
    controller.load_model(save_path)
    controller.eval_mode()
    
    apply_temp = True if controller.temp > 0 else False
    with torch.no_grad():
        r, loss_rall = controller.evaluate_with_list_id(list_image_id_test, list_test, para_standardize, apply_temp)
        r1i, r5i, r10i, r1t, r5t, r10t = r
        
    info_txt = f"R1i: {r1i}\nR5i: {r5i}\nR10i: {r10i}\n"
    info_txt += f"R1t: {r1t}\nR5t: {r5t}\nR10t: {r10t}\n"
    info_txt += f"Ri: {r1i+r5i+r10i}\nRt: {r1t+r5t+r10t}\n"
    info_txt += f"Rall: {r1i+r5i+r10i+r1t+r5t+r10t}\n"
    info_txt += f"LoRe: {loss_rall}\n"
    write_to_file(f"{model_type}/{config['out_dir']}/{config_name}/TestReport.log", info_txt)     
    print(info_txt)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='HADA_m/Config/C5.yml', help='yml file of the config')
    # parser.add_argument('-md', '--model_type', type=str, default='LiFu_m', help='structure of the model')
    parser.add_argument('-rm', '--run_mode', type=str, default='train', help='train: train and test\ntest: only test')
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    print(f"CONFIG: {CONFIG_PATH.split('/')[-1]}")
    if args.run_mode == 'train':
        run_train(args)
        run_evaluate(args)
    if args.run_mode == 'test':
        run_evaluate(args)