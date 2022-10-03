import json
import Utils as ut
from Controller import Controller
import joblib
import argparse
import numpy as np
from Retrieval_Utils import i2t, t2i, evaluate_recall
import torch
import mlflow

JSON_TRAIN = '../JSON/train_matching_image_preprocessed_cap.json'
JSON_VAL = '../JSON/val_matching_image_preprocessed_cap.json'
JSON_TEST = '../JSON/test_matching_image_preprocessed_cap.json'

# try:
#     para_normalize = joblib.load('../normalize_para.joblib')
#     print("Found Normalize Para")
# except:
#     print("NOT Found Normalize Para")
#     para_normalize = None
    
try:
    para_standardize = joblib.load('../standardize_each.joblib')
    # para_standardize['a'] = para_standardize['all']
    # para_standardize['d'] = para_standardize['all']
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

list_train = []
list_val = []
list_test = []
for x in list_image_id_train:
    cap_data = json_train[x]
    ncap = len(cap_data)
    for y in range(ncap):
        fic = f"{x}_{y}"
        list_train.append(fic)
        
for x in list_image_id_val:
    cap_data = json_val[x]
    ncap = len(cap_data)
    for y in range(ncap):
        fic = f"{x}_{y}"
        list_val.append(fic)
    
for x in list_image_id_test:
    cap_data = json_test[x]
    ncap = len(cap_data)
    for y in range(ncap):
        fic = f"{x}_{y}"
        list_test.append(fic)
        
# dataset_test = ut.Image_Caption_Dataset(list_img_id_cap_id=list_test, 
#                                         albef_dir=ALBEF_OUTPUT, dot_dir=DOT_OUTPUT, 
#                                         para_normalize=para_normalize,
#                                         para_standardize=para_standardize)
# dataloader_test = ut.make_dataloader(dataset_test, branch='both', batch_size=config['batch_size'], shuffle=False)

def run_train(config_path):
    print(f"RUN TRAIN")
    config_name = config_path.split('/')[-1][:-4]
    config = ut.load_config(config_path)
    
    niters = int(int(np.ceil(145000 / config['batch_size'])))
    if config['Tmax'] > 0:
        config['Tmax'] = config['Tmax'] * niters
        
    controller = Controller(config)
    total_para = controller.count_parameters()
    print(f"Trainable Paras: {total_para}")
    controller.train(para_standardize=para_standardize,
                     train_img_id=list_image_id_train, 
                     train_cap_id=list_train,
                     val_img_id=list_image_id_val, 
                     val_cap_id=list_val,
                     num_epoch=config['num_epoch'], model_name=config_name)

def run_evaluate(config_path):
    print(f"PERFORM EVALUATE")
    config_name = config_path.split('/')[-1][:-4]
    config = ut.load_config(config_path)
    save_path = f"{config['out_dir']}/{config_name}/best.pth.tar"
    
    controller = Controller(config)
    controller.load_model(save_path)
    
    controller.eval_mode()
    
    apply_temp = True if controller.temp > 0 else False
    with torch.no_grad():
        r, loss_rall = controller.evaluate_with_list_id(list_image_id_test, list_test, para_standardize, apply_temp)
        r1i, r5i, r10i, r1t, r5t, r10t = r
        
    info_txt = f"R1i: {np.round(r1i,4)}\nR5i: {np.round(r5i,4)}\nR10i: {np.round(r10i,4)}\n"
    info_txt += f"R1t: {np.round(r1t,4)}\nR5t: {np.round(r5t,4)}\nR10t: {np.round(r10t,4)}\n"
    info_txt += f"Ri: {np.round(r1i+r5i+r10i,4)}\nRt: {np.round(r1t+r5t+r10t,4)}\n"
    info_txt += f"Rall: {np.round(r1i+r5i+r10i+r1t+r5t+r10t,4)}\n"
    info_txt += f"LoRe: {np.round(loss_rall, 4)}\n"
    ut.write_to_file(f"{config['out_dir']}/{config_name}/TestReport.log", info_txt)     
    print(info_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='Config/C1.yml', help='yml file of the config')
    parser.add_argument('-rm', '--run_mode', type=str, default='train', help='train: train and test\ntest: only test')
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    # TOPK = args.top_k
    print(f"CONFIG: {CONFIG_PATH.split('/')[-1]}")
    if args.run_mode == 'train':
        run_train(config_path=CONFIG_PATH)
        run_evaluate(config_path=CONFIG_PATH)
        # pre_rank(config_path=CONFIG_PATH, topk=TOPK)
    if args.run_mode == 'test':
        run_evaluate(config_path=CONFIG_PATH)
        # pre_rank(config_path=CONFIG_PATH, topk=TOPK)