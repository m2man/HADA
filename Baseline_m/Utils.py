import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import random
import torch
import itertools
import yaml
import torch.nn.functional as F

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    
def do_normalize(data, para_min, para_max):
    normed = (data - para_min) / (para_max - para_min)
    return normed

def do_standardize(data, mean, std):
    normed = (data - mean) / std
    return normed

def load_config(filename):
    with open(filename) as file:
        config_dict= yaml.safe_load(file)
    return config_dict

def write_to_file(filepath='text.txt', content=''):
    with open(filepath, "a") as f_log:
        f_log.write(content)

def create_index_from_2_list(list_1, list_2, dual_index=False, self_loop=False):
    first = np.repeat(list_1, len(list_2))
    second = np.tile(list_2, len(list_1))
    result = np.asarray([first, second])
    if dual_index:
        first = np.repeat(list_2, len(list_1))
        second = np.tile(list_1, len(list_2))
        result = np.concatenate((result, np.asarray([first, second])), axis=1)
    if self_loop:
        list_all = list_1 + list_2
        result = np.concatenate((result, np.asarray([list_all, list_all])), axis=1)
    return result

def create_index(start_idx, end_idx, self_loop=True):
    # create index (2, (end_idx-start_idx)**2) 
    n = end_idx - start_idx + 1
    r = [x for x in range(start_idx, end_idx+1)]
    first = np.repeat(r, n)
    second = np.tile(r, n)
    result = np.asarray([first, second])
    if not self_loop:
        result = result[:,np.where((result[0] - result[1]) != 0)[0]]
    return result

def create_edge_index(list_num_nodes, self_loop=False):
    batch_size = len(list_num_nodes)
    count = 0
    for idx in range(batch_size):
        this_num_nodes = list_num_nodes[idx]
        if idx == 0:
            edge_indices = create_index(count, count+this_num_nodes-1, self_loop=self_loop)
        else:
            edge_indices = np.concatenate((edge_indices,
                                           create_index(count, count+this_num_nodes-1, self_loop=self_loop)),
                                           axis=1)
        count += this_num_nodes 
    edge_indices = torch.tensor(edge_indices) # (2, total edges)                                       
    return edge_indices

class Image_Caption_Dataset(Dataset):
    def __init__(self, list_img_id_cap_id, config, para_standardize=None):        
        self.albef_dir = config['albef_dir']
        self.dot_dir = config['dot_dir']
        self.albef_augment_dir = config['albef_augment_dir']
        self.list_img_id_cap_id = list_img_id_cap_id.copy()
        self.para_standardize = para_standardize
        self.seed = 1509
        self.util_norm = config['util_norm']
        
    def __len__(self):
        return len(self.list_img_id_cap_id)
    
    def shuffle(self, seed):
        # do this at every epoch
        self.seed = seed
        random.Random(seed).shuffle(self.list_img_id_cap_id)
        
    def __getitem__(self, index):    
        img_id_cap_id = self.list_img_id_cap_id[index]
        img_id, cap_id = img_id_cap_id.split('_')
        cap_id = int(cap_id)
        # ALBEF
        if self.albef_augment_dir is not None:
            # random.seed(self.seed + 1509 + index)
            using_augment = random.random() > 0.25
            if using_augment:
                idx_augment = random.randint(0,2)
                albef_img = np.load(f"{self.albef_augment_dir}/images/{img_id}_{idx_augment}.npz")
            else:
                albef_img = np.load(f"{self.albef_dir}/images/{img_id}.npz")
        else:
            albef_img = np.load(f"{self.albef_dir}/images/{img_id}.npz")
        albef_cap = np.load(f"{self.albef_dir}/texts/{img_id}.npz")
        albef_img_cls = albef_img['proj_embed'] # the projected CLS (256,)
        albef_cap_cls = albef_cap['proj_embed'][cap_id] # (256,)
        # DOT
        dot_img = np.load(f"{self.dot_dir}/images/{img_id}.npz")
        dot_cap = np.load(f"{self.dot_dir}/texts/{img_id}.npz",allow_pickle=True)
        # dot_img_feat = dot_img['feat'][1:] # the CLS is projected with additional layer (n_obj, 768)
        dot_img_cls = dot_img['proj_embed'] # the projected CLS (768,)
        dot_cap_cls = dot_cap['proj_embed'][cap_id] # (768,)
        # Convert to tensor
        albef_cap_cls = torch.tensor(albef_cap_cls)
        albef_img_cls = torch.tensor(albef_img_cls)
        dot_cap_cls = torch.tensor(dot_cap_cls)
        dot_img_cls = torch.tensor(dot_img_cls)
        albef_cap_cls_ori = torch.clone(albef_cap_cls)
        albef_img_cls_ori = torch.clone(albef_img_cls)
        dot_cap_cls_ori = torch.clone(dot_cap_cls)
        dot_img_cls_ori = torch.clone(dot_img_cls)
        # Normalize
        if self.util_norm:
            albef_cap_cls = F.normalize(albef_cap_cls.view(1,-1)).squeeze()
            albef_img_cls = F.normalize(albef_img_cls.view(1,-1)).squeeze()
            dot_cap_cls = F.normalize(dot_cap_cls.view(1,-1)).squeeze()
            dot_img_cls = F.normalize(dot_img_cls.view(1,-1)).squeeze()
        # Normalize data
        if self.para_standardize is not None:
            if 'a_i' in list(self.para_standardize.keys()):
                albef_cap_cls = do_standardize(albef_cap_cls, self.para_standardize['a_t']['mean'].item(), self.para_standardize['a_t']['std'].item())
                albef_img_cls = do_standardize(albef_img_cls, self.para_standardize['a_i']['mean'].item(), self.para_standardize['a_i']['std'].item())
                dot_cap_cls = do_standardize(dot_cap_cls, self.para_standardize['d_t']['mean'].item(), self.para_standardize['d_t']['std'].item())
                dot_img_cls = do_standardize(dot_img_cls, self.para_standardize['d_i']['mean'].item(), self.para_standardize['d_i']['std'].item())
        
        albef_img_cls = albef_img_cls.reshape(1,-1)
        dot_img_cls = dot_img_cls.reshape(1,-1)
        albef_cap_cls = albef_cap_cls.reshape(1,-1)
        dot_cap_cls = dot_cap_cls.reshape(1,-1)
        
        albef_img_cls_ori = albef_img_cls_ori.reshape(1,-1)
        dot_img_cls_ori = dot_img_cls_ori.reshape(1,-1)
        albef_cap_cls_ori = albef_cap_cls_ori.reshape(1,-1)
        dot_cap_cls_ori = dot_cap_cls_ori.reshape(1,-1)
                
        dict_return = {}
        dict_return['img'] = {'cls_albef': albef_img_cls, # (1,256) 
                              'cls_dot': dot_img_cls, # (1, 768)
                              'cls_albef_ori': albef_img_cls_ori, 
                              'cls_dot_ori': dot_img_cls_ori} 
        
        dict_return['cap'] = {'cls_albef': albef_cap_cls, # (1,256) 
                              'cls_dot': dot_cap_cls, # (1, 768)
                              'cls_albef_ori': albef_cap_cls_ori, 
                              'cls_dot_ori': dot_cap_cls_ori}
        dict_return['id'] = f"{img_id}_{cap_id}"
        return dict_return
    
class Caption_Dataset(Dataset):
    def __init__(self, list_img_id_cap_id, config, para_standardize=None):        
        self.albef_dir = config['albef_dir']
        self.dot_dir = config['dot_dir']
        self.list_img_id_cap_id = list_img_id_cap_id
        self.para_standardize = para_standardize
        self.util_norm = config['util_norm']
        
    def __len__(self):
        return len(self.list_img_id_cap_id)
    
    def shuffle(self, seed):
        # do this at every epoch
        random.Random(seed).shuffle(self.list_img_id_cap_id)
        
    def __getitem__(self, index):    
        img_id_cap_id = self.list_img_id_cap_id[index]
        img_id, cap_id = img_id_cap_id.split('_')
        cap_id = int(cap_id)
        # ALBEF
        albef_cap = np.load(f"{self.albef_dir}/texts/{img_id}.npz")
        albef_cap_cls = albef_cap['proj_embed'][cap_id] # (256,)
        # DOT
        dot_cap = np.load(f"{self.dot_dir}/texts/{img_id}.npz",allow_pickle=True)
        dot_cap_cls = dot_cap['proj_embed'][cap_id] # (768,)
        # Convert to tensor
        albef_cap_cls = torch.tensor(albef_cap_cls)
        dot_cap_cls = torch.tensor(dot_cap_cls)
        albef_cap_cls_ori = torch.clone(albef_cap_cls)
        dot_cap_cls_ori = torch.clone(dot_cap_cls)
        # Normalize
        if self.util_norm:
            albef_cap_cls = F.normalize(albef_cap_cls.view(1,-1)).squeeze()
            dot_cap_cls = F.normalize(dot_cap_cls.view(1,-1)).squeeze()
        if self.para_standardize is not None:
            if 'a_i' in list(self.para_standardize.keys()):
                albef_cap_cls = do_standardize(albef_cap_cls, self.para_standardize['a_t']['mean'].item(), self.para_standardize['a_t']['std'].item())
                dot_cap_cls = do_standardize(dot_cap_cls, self.para_standardize['d_t']['mean'].item(), self.para_standardize['d_t']['std'].item())
        albef_cap_cls = albef_cap_cls.reshape(1,-1)
        dot_cap_cls = dot_cap_cls.reshape(1,-1)
        albef_cap_cls_ori = albef_cap_cls_ori.reshape(1,-1)
        dot_cap_cls_ori = dot_cap_cls_ori.reshape(1,-1)
        
        dict_return = {}
        dict_return['cap'] = {'cls_albef': albef_cap_cls, # (1,256) 
                              'cls_dot': dot_cap_cls, # (1, 768)
                              'cls_albef_ori': albef_cap_cls_ori, 
                              'cls_dot_ori': dot_cap_cls_ori}
        dict_return['id'] = f"{img_id}_{cap_id}"
        return dict_return

class Image_Dataset(Dataset):
    def __init__(self, list_img_id, config, para_normalize=None, para_standardize=None):        
        self.albef_dir = config['albef_dir']
        self.dot_dir = config['dot_dir']
        self.list_img_id = list_img_id
        self.para_standardize = para_standardize
        self.util_norm = config['util_norm']
        
    def __len__(self):
        return len(self.list_img_id)
    
    def shuffle(self, seed):
        # do this at every epoch
        random.Random(seed).shuffle(self.list_img_id)
        
    def __getitem__(self, index):    
        img_id = self.list_img_id[index]
        # ALBEF
        albef_img = np.load(f"{self.albef_dir}/images/{img_id}.npz")
        albef_img_cls = albef_img['proj_embed'] # the projected CLS (256,)
        # DOT
        dot_img = np.load(f"{self.dot_dir}/images/{img_id}.npz")
        dot_img_cls = dot_img['proj_embed'] # the projected CLS (768,)
        # Convert to tensor
        albef_img_cls = torch.tensor(albef_img_cls)
        dot_img_cls = torch.tensor(dot_img_cls)
        albef_img_cls_ori = torch.clone(albef_img_cls)
        dot_img_cls_ori = torch.clone(dot_img_cls)
        # Normalize
        if self.util_norm:
            albef_img_cls = F.normalize(albef_img_cls.view(1,-1)).squeeze()
            dot_img_cls = F.normalize(dot_img_cls.view(1,-1)).squeeze()
        if self.para_standardize is not None:
            if 'a_i' in list(self.para_standardize.keys()):
                albef_img_cls = do_standardize(albef_img_cls, self.para_standardize['a_i']['mean'].item(), self.para_standardize['a_i']['std'].item())
                dot_img_cls = do_standardize(dot_img_cls, self.para_standardize['d_i']['mean'].item(), self.para_standardize['d_i']['std'].item())
        
        albef_img_cls = albef_img_cls.reshape(1,-1)
        dot_img_cls = dot_img_cls.reshape(1,-1)
        albef_img_cls_ori = albef_img_cls_ori.reshape(1,-1)
        dot_img_cls_ori = dot_img_cls_ori.reshape(1,-1)
        
        dict_return = {}
        dict_return['img'] = {'cls_albef': albef_img_cls, # (1,256) 
                              'cls_dot': dot_img_cls, # (1, 768)
                              'cls_albef_ori': albef_img_cls_ori, 
                              'cls_dot_ori': dot_img_cls_ori}
        
        dict_return['id'] = f"{img_id}"
        return dict_return
    
def collate_function_both(batch):
    cap_cls_albef = torch.tensor(())
    cap_cls_dot = torch.tensor(())
    img_cls_albef = torch.tensor(())
    img_cls_dot = torch.tensor(())  
    cap_cls_albef_ori = torch.tensor(())
    cap_cls_dot_ori = torch.tensor(())
    img_cls_albef_ori = torch.tensor(())
    img_cls_dot_ori = torch.tensor(())
    list_id = []
    for x in batch:
        img_cls_albef = torch.cat((img_cls_albef, x['img']['cls_albef']), dim=0)
        img_cls_dot = torch.cat((img_cls_dot, x['img']['cls_dot']), dim=0)
        img_cls_albef_ori = torch.cat((img_cls_albef_ori, x['img']['cls_albef_ori']), dim=0)
        img_cls_dot_ori = torch.cat((img_cls_dot_ori, x['img']['cls_dot_ori']), dim=0)
        cap_cls_albef = torch.cat((cap_cls_albef, x['cap']['cls_albef']), dim=0)
        cap_cls_dot = torch.cat((cap_cls_dot, x['cap']['cls_dot']), dim=0)
        cap_cls_albef_ori = torch.cat((cap_cls_albef_ori, x['cap']['cls_albef_ori']), dim=0)
        cap_cls_dot_ori = torch.cat((cap_cls_dot_ori, x['cap']['cls_dot_ori']), dim=0)
        list_id.append(x['id'])        
    bs = len(list_id)
    img_dict = {'cls_albef': img_cls_albef, 'cls_dot': img_cls_dot, 
                'cls_albef_ori': img_cls_albef_ori, 'cls_dot_ori': img_cls_dot_ori}
    cap_dict = {'cls_albef': cap_cls_albef, 'cls_dot': cap_cls_dot,
                'cls_albef_ori': cap_cls_albef_ori, 'cls_dot_ori': cap_cls_dot_ori}
    list_id = torch.tensor([[int(x.split('_')[0]) for x in list_id]]).reshape(-1,1)
    return img_dict, cap_dict, list_id

def collate_function_img(batch):   
    img_cls_albef = torch.tensor(())
    img_cls_dot = torch.tensor(())
    img_cls_albef_ori = torch.tensor(())
    img_cls_dot_ori = torch.tensor(())
    list_id = []
    for x in batch:
        img_cls_albef = torch.cat((img_cls_albef, x['img']['cls_albef']), dim=0)
        img_cls_dot = torch.cat((img_cls_dot, x['img']['cls_dot']), dim=0)
        img_cls_albef_ori = torch.cat((img_cls_albef_ori, x['img']['cls_albef_ori']), dim=0)
        img_cls_dot_ori = torch.cat((img_cls_dot_ori, x['img']['cls_dot_ori']), dim=0)
        list_id.append(x['id'])   
    bs = len(list_id)
    img_dict = {'cls_albef': img_cls_albef, 'cls_dot': img_cls_dot,
                'cls_albef_ori': img_cls_albef_ori, 'cls_dot_ori': img_cls_dot_ori}
    return img_dict, list_id
  
def collate_function_cap(batch):
    cap_cls_albef = torch.tensor(())
    cap_cls_dot = torch.tensor(())
    cap_cls_albef_ori = torch.tensor(())
    cap_cls_dot_ori = torch.tensor(())
    list_id = []
    for x in batch:
        cap_cls_albef = torch.cat((cap_cls_albef, x['cap']['cls_albef']), dim=0)
        cap_cls_dot = torch.cat((cap_cls_dot, x['cap']['cls_dot']), dim=0)
        cap_cls_albef_ori = torch.cat((cap_cls_albef_ori, x['cap']['cls_albef_ori']), dim=0)
        cap_cls_dot_ori = torch.cat((cap_cls_dot_ori, x['cap']['cls_dot_ori']), dim=0)
        list_id.append(x['id'])    
    bs = len(list_id)
    cap_dict = {'cls_albef': cap_cls_albef, 'cls_dot': cap_cls_dot,
                'cls_albef_ori': cap_cls_albef_ori, 'cls_dot_ori': cap_cls_dot_ori}
    return cap_dict, list_id

def make_dataloader(dataset, branch='both', **args):
    if branch == 'both':
        return DataLoader(dataset, collate_fn=collate_function_both, **args)
    if branch == 'img' or branch == 'image':
        return DataLoader(dataset, collate_fn=collate_function_img, **args)
    if branch == 'cap' or branch == 'txt':
        return DataLoader(dataset, collate_fn=collate_function_cap, **args)

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    min_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            min_grads.append(p.grad.abs().min())
    
    return layers, min_grads, ave_grads, max_grads