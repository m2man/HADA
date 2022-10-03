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
        self.wffx = config['wffx']
        self.wfc = config['wfc']
        self.wfcx = config['wfcx']
        self.wcc = config['wcc']
        self.self_loop = True
        self.directed_graph = config['directed_graph']
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
        dot_img = np.load(f"{self.dot_dir}/images/{img_id}.npz")
        dot_cap = np.load(f"{self.dot_dir}/texts/{img_id}.npz",allow_pickle=True)
        # dot_img_feat = dot_img['feat'][1:] # the CLS is projected with additional layer (n_obj, 768)
        dot_img_feat = dot_img['feat'][0:] # the CLS is projected with additional layer (1_cls+n_obj, 768)
        dot_img_cls = dot_img['proj_embed'] # the projected CLS (768,)
        dot_cap_feat = dot_cap['feat'][cap_id] # include CLS (n_len, 768)
        dot_cap_att = dot_cap['att'][cap_id] # (n_len)
        len_cap = sum(dot_cap_att) # include the CLS
        # dot_cap_feat = dot_cap_feat[1:len_cap] # (len_cap-1, 768)
        dot_cap_feat = dot_cap_feat[0:len_cap] # (len_cap, 768) # --> 1st element is CLS (before projected as cap_cls)
        dot_cap_cls = dot_cap['proj_embed'][cap_id] # (768,)
        # Convert to tensor
        albef_cap_feat = torch.tensor(albef_cap_feat)
        albef_cap_cls = torch.tensor(albef_cap_cls)
        albef_img_feat = torch.tensor(albef_img_feat)
        albef_img_cls = torch.tensor(albef_img_cls)
        dot_cap_feat = torch.tensor(dot_cap_feat)
        dot_cap_cls = torch.tensor(dot_cap_cls)
        dot_img_feat = torch.tensor(dot_img_feat)
        dot_img_cls = torch.tensor(dot_img_cls)
        # Normalize
        if self.util_norm:
            albef_cap_feat = F.normalize(albef_cap_feat, dim=1)
            albef_cap_cls = F.normalize(albef_cap_cls.view(1,-1)).squeeze()
            albef_img_feat = F.normalize(albef_img_feat, dim=1)
            albef_img_cls = F.normalize(albef_img_cls.view(1,-1)).squeeze()
            dot_cap_feat = F.normalize(dot_cap_feat, dim=1)
            dot_cap_cls = F.normalize(dot_cap_cls.view(1,-1)).squeeze()
            dot_img_feat = F.normalize(dot_img_feat, dim=1)
            dot_img_cls = F.normalize(dot_img_cls.view(1,-1)).squeeze()
        # Normalize data
        if self.para_standardize is not None:
            if 'a_i' in list(self.para_standardize.keys()):
                albef_cap_feat = do_standardize(albef_cap_feat, self.para_standardize['a_t']['mean'].item(), self.para_standardize['a_t']['std'].item())
                # albef_cap_cls = do_standardize(albef_cap_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_img_feat = do_standardize(albef_img_feat, self.para_standardize['a_i']['mean'].item(), self.para_standardize['a_i']['std'].item())
                # albef_img_cls = do_standardize(albef_img_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                dot_cap_feat = do_standardize(dot_cap_feat, self.para_standardize['d_t']['mean'].item(), self.para_standardize['d_t']['std'].item())
                # dot_cap_cls = do_standardize(dot_cap_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_img_feat = do_standardize(dot_img_feat, self.para_standardize['d_i']['mean'].item(), self.para_standardize['d_i']['std'].item())
                # dot_img_cls = do_standardize(dot_img_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
            else:
                albef_cap_feat = do_standardize(albef_cap_feat, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_cap_cls = do_standardize(albef_cap_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_img_feat = do_standardize(albef_img_feat, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_img_cls = do_standardize(albef_img_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                dot_cap_feat = do_standardize(dot_cap_feat, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_cap_cls = do_standardize(dot_cap_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_img_feat = do_standardize(dot_img_feat, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_img_cls = do_standardize(dot_img_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
        
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
        f1_c1 = create_index_from_2_list(f1, c1, dual_index=not self.directed_graph, self_loop=self.self_loop) # (2, n_edge)
        f2_c2 = create_index_from_2_list(f2, c2, dual_index=not self.directed_graph, self_loop=self.self_loop)
        f2_c1 = create_index_from_2_list(f2, c1, dual_index=not self.directed_graph, self_loop=False)
        f1_c2 = create_index_from_2_list(f1, c2, dual_index=not self.directed_graph, self_loop=False)
        c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
        f1_f2 = create_index_from_2_list(f1, f2, dual_index=True, self_loop=False)
        edge_index = np.concatenate((f1_c1, f2_c2, c1_c2, f1_c2, f2_c1, f1_f2), axis=1)
        # create edge attr
        f1_c1_attr = [self.wfc for x in range(f1_c1.shape[1])]
        f2_c2_attr = [self.wfc for x in range(f2_c2.shape[1])]
        if self.self_loop:
            f1_c1_attr[-1] = self.wcc
            f2_c2_attr[-1] = self.wcc
        c1_c2_attr = [self.wcc for x in range(c1_c2.shape[1])]
        f1_c2_attr = [self.wfcx for x in range(f1_c2.shape[1])]
        f2_c1_attr = [self.wfcx for x in range(f2_c1.shape[1])]
        f1_f2_attr = [self.wffx for x in range(f1_f2.shape[1])]
        edge_attr = np.array(f1_c1_attr + f2_c2_attr + c1_c2_attr + f1_c2_attr + f2_c1_attr + f1_f2_attr)
        # drop ffx or not
        drop_idx = np.where(edge_attr==-1)[0]
        img_edge_attr = torch.tensor(np.delete(edge_attr, drop_idx), dtype=torch.float)
        img_edge_index = torch.tensor(np.delete(edge_index, drop_idx, 1))
        
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
        f1_c1 = create_index_from_2_list(f1, c1, dual_index=not self.directed_graph, self_loop=self.self_loop) # (2, n_edge)
        f2_c2 = create_index_from_2_list(f2, c2, dual_index=not self.directed_graph, self_loop=self.self_loop)
        f2_c1 = create_index_from_2_list(f2, c1, dual_index=not self.directed_graph, self_loop=False)
        f1_c2 = create_index_from_2_list(f1, c2, dual_index=not self.directed_graph, self_loop=False)
        c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
        f1_f2 = create_index_from_2_list(f1, f2, dual_index=True, self_loop=False)
        edge_index = np.concatenate((f1_c1, f2_c2, c1_c2, f1_c2, f2_c1, f1_f2), axis=1)
        # create edge attr
        f1_c1_attr = [self.wfc for x in range(f1_c1.shape[1])]
        f2_c2_attr = [self.wfc for x in range(f2_c2.shape[1])]
        if self.self_loop:
            f1_c1_attr[-1] = self.wcc
            f2_c2_attr[-1] = self.wcc
        c1_c2_attr = [self.wcc for x in range(c1_c2.shape[1])]
        f1_c2_attr = [self.wfcx for x in range(f1_c2.shape[1])]
        f2_c1_attr = [self.wfcx for x in range(f2_c1.shape[1])]
        f1_f2_attr = [self.wffx for x in range(f1_f2.shape[1])]
        edge_attr = np.array(f1_c1_attr + f2_c2_attr + c1_c2_attr + f1_c2_attr + f2_c1_attr + f1_f2_attr)
        # drop ffx or not
        drop_idx = np.where(edge_attr==-1)[0]
        cap_edge_attr = torch.tensor(np.delete(edge_attr, drop_idx), dtype=torch.float)
        cap_edge_index = torch.tensor(np.delete(edge_index, drop_idx, 1))
        
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
        return dict_return
    
class Caption_Dataset(Dataset):
    def __init__(self, list_img_id_cap_id, config, para_standardize=None):        
        self.albef_dir = config['albef_dir']
        self.dot_dir = config['dot_dir']
        self.wffx = config['wffx']
        self.wfc = config['wfc']
        self.wfcx = config['wfcx']
        self.wcc = config['wcc']
        self.self_loop = True
        self.directed_graph = config['directed_graph']
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
        albef_cap_feat = albef_cap['feat'][cap_id] # 30, 768
        albef_cap_cls = albef_cap['proj_embed'][cap_id] # (256,)
        albef_cap_att = albef_cap['att'][cap_id] # (30,) binary vector indicate len of a caption 1, 1, 1, 0, 0...
        len_cap = sum(albef_cap_att) # include the CLS
        albef_cap_feat = albef_cap_feat[0:len_cap] # --> 1st element is CLS (before projected as cap_cls) (len_cap, 768)
        # DOT
        dot_cap = np.load(f"{self.dot_dir}/texts/{img_id}.npz",allow_pickle=True)
        dot_cap_feat = dot_cap['feat'][cap_id] # include CLS (n_len, 768)
        dot_cap_att = dot_cap['att'][cap_id] # (n_len)
        len_cap = sum(dot_cap_att) # include the CLS
        dot_cap_feat = dot_cap_feat[0:len_cap] # (len_cap, 768) # --> 1st element is CLS (before projected as cap_cls)
        dot_cap_cls = dot_cap['proj_embed'][cap_id] # (768,)
        # Convert to tensor
        albef_cap_feat = torch.tensor(albef_cap_feat)
        albef_cap_cls = torch.tensor(albef_cap_cls)
        dot_cap_feat = torch.tensor(dot_cap_feat)
        dot_cap_cls = torch.tensor(dot_cap_cls)
        # Normalize
        if self.util_norm:
            albef_cap_feat = F.normalize(albef_cap_feat, dim=1)
            albef_cap_cls = F.normalize(albef_cap_cls.view(1,-1)).squeeze()
            dot_cap_feat = F.normalize(dot_cap_feat, dim=1)
            dot_cap_cls = F.normalize(dot_cap_cls.view(1,-1)).squeeze()
        if self.para_standardize is not None:
            if 'a_i' in list(self.para_standardize.keys()):
                albef_cap_feat = do_standardize(albef_cap_feat, self.para_standardize['a_t']['mean'].item(), self.para_standardize['a_t']['std'].item())
                # albef_cap_cls = do_standardize(albef_cap_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                dot_cap_feat = do_standardize(dot_cap_feat, self.para_standardize['d_t']['mean'].item(), self.para_standardize['d_t']['std'].item())
                # dot_cap_cls = do_standardize(dot_cap_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
            else:
                albef_cap_feat = do_standardize(albef_cap_feat, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_cap_cls = do_standardize(albef_cap_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                dot_cap_feat = do_standardize(dot_cap_feat, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_cap_cls = do_standardize(dot_cap_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
        
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
        f1_c1 = create_index_from_2_list(f1, c1, dual_index=not self.directed_graph, self_loop=self.self_loop) # (2, n_edge)
        f2_c2 = create_index_from_2_list(f2, c2, dual_index=not self.directed_graph, self_loop=self.self_loop)
        f2_c1 = create_index_from_2_list(f2, c1, dual_index=not self.directed_graph, self_loop=False)
        f1_c2 = create_index_from_2_list(f1, c2, dual_index=not self.directed_graph, self_loop=False)
        c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
        f1_f2 = create_index_from_2_list(f1, f2, dual_index=True, self_loop=False)
        edge_index = np.concatenate((f1_c1, f2_c2, c1_c2, f1_c2, f2_c1, f1_f2), axis=1)
        # create edge attr
        f1_c1_attr = [self.wfc for x in range(f1_c1.shape[1])]
        f2_c2_attr = [self.wfc for x in range(f2_c2.shape[1])]
        if self.self_loop:
            f1_c1_attr[-1] = self.wcc
            f2_c2_attr[-1] = self.wcc
        c1_c2_attr = [self.wcc for x in range(c1_c2.shape[1])]
        f1_c2_attr = [self.wfcx for x in range(f1_c2.shape[1])]
        f2_c1_attr = [self.wfcx for x in range(f2_c1.shape[1])]
        f1_f2_attr = [self.wffx for x in range(f1_f2.shape[1])]
        edge_attr = np.array(f1_c1_attr + f2_c2_attr + c1_c2_attr + f1_c2_attr + f2_c1_attr + f1_f2_attr)
        # drop ffx or not
        drop_idx = np.where(edge_attr==-1)[0]
        cap_edge_attr = torch.tensor(np.delete(edge_attr, drop_idx), dtype=torch.float)
        cap_edge_index = torch.tensor(np.delete(edge_index, drop_idx, 1))
        
        albef_cap_cls = albef_cap_cls.reshape(1,-1)
        dot_cap_cls = dot_cap_cls.reshape(1,-1)
        
        dict_return = {}
        dict_return['cap'] = {'node_albef': albef_cap_feat, # (1+len_cap)
                              'node_dot': dot_cap_feat, # (1+len_cap, 768)
                              'edge_index': cap_edge_index, #(2, len_edge)
                              'edge_attr': cap_edge_attr, # (len_edge)
                              'cls_albef': albef_cap_cls, # (1,256) 
                              'cls_dot': dot_cap_cls} # (1, 768)
        dict_return['id'] = f"{img_id}_{cap_id}"
        return dict_return

class Image_Dataset(Dataset):
    def __init__(self, list_img_id, config, para_normalize=None, para_standardize=None):        
        self.albef_dir = config['albef_dir']
        self.dot_dir = config['dot_dir']
        self.wffx = config['wffx']
        self.wfc = config['wfc']
        self.wfcx = config['wfcx']
        self.wcc = config['wcc']
        self.self_loop = True
        self.directed_graph = config['directed_graph']
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
        albef_img_feat = albef_img['feat'][0:] # the CLS is projected with additional layer (1_cls+576, 768)
        albef_img_cls = albef_img['proj_embed'] # the projected CLS (256,)
        # DOT
        dot_img = np.load(f"{self.dot_dir}/images/{img_id}.npz")
        dot_img_feat = dot_img['feat'][0:] # the CLS is projected with additional layer (1_cls+n_obj, 768)
        dot_img_cls = dot_img['proj_embed'] # the projected CLS (768,)
        # Convert to tensor
        albef_img_feat = torch.tensor(albef_img_feat)
        albef_img_cls = torch.tensor(albef_img_cls)
        dot_img_feat = torch.tensor(dot_img_feat)
        dot_img_cls = torch.tensor(dot_img_cls)
        # Normalize
        if self.util_norm:
            albef_img_feat = F.normalize(albef_img_feat, dim=1)
            albef_img_cls = F.normalize(albef_img_cls.view(1,-1)).squeeze()
            dot_img_feat = F.normalize(dot_img_feat, dim=1)
            dot_img_cls = F.normalize(dot_img_cls.view(1,-1)).squeeze()
        if self.para_standardize is not None:
            if 'a_i' in list(self.para_standardize.keys()):
                albef_img_feat = do_standardize(albef_img_feat, self.para_standardize['a_i']['mean'].item(), self.para_standardize['a_i']['std'].item())
                # albef_img_cls = do_standardize(albef_img_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                dot_img_feat = do_standardize(dot_img_feat, self.para_standardize['d_i']['mean'].item(), self.para_standardize['d_i']['std'].item())
                # dot_img_cls = do_standardize(dot_img_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
            else:
                albef_img_feat = do_standardize(albef_img_feat, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_img_cls = do_standardize(albef_img_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                dot_img_feat = do_standardize(dot_img_feat, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_img_cls = do_standardize(dot_img_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())

        
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
        f1_c1 = create_index_from_2_list(f1, c1, dual_index=not self.directed_graph, self_loop=self.self_loop) # (2, n_edge)
        f2_c2 = create_index_from_2_list(f2, c2, dual_index=not self.directed_graph, self_loop=self.self_loop)
        f2_c1 = create_index_from_2_list(f2, c1, dual_index=not self.directed_graph, self_loop=False)
        f1_c2 = create_index_from_2_list(f1, c2, dual_index=not self.directed_graph, self_loop=False)
        c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
        f1_f2 = create_index_from_2_list(f1, f2, dual_index=True, self_loop=False)
        edge_index = np.concatenate((f1_c1, f2_c2, c1_c2, f1_c2, f2_c1, f1_f2), axis=1)
        # create edge attr
        f1_c1_attr = [self.wfc for x in range(f1_c1.shape[1])]
        f2_c2_attr = [self.wfc for x in range(f2_c2.shape[1])]
        if self.self_loop:
            f1_c1_attr[-1] = self.wcc
            f2_c2_attr[-1] = self.wcc
        c1_c2_attr = [self.wcc for x in range(c1_c2.shape[1])]
        f1_c2_attr = [self.wfcx for x in range(f1_c2.shape[1])]
        f2_c1_attr = [self.wfcx for x in range(f2_c1.shape[1])]
        f1_f2_attr = [self.wffx for x in range(f1_f2.shape[1])]
        edge_attr = np.array(f1_c1_attr + f2_c2_attr + c1_c2_attr + f1_c2_attr + f2_c1_attr + f1_f2_attr)
        # drop ffx or not
        drop_idx = np.where(edge_attr==-1)[0]
        img_edge_attr = torch.tensor(np.delete(edge_attr, drop_idx), dtype=torch.float)
        img_edge_index = torch.tensor(np.delete(edge_index, drop_idx, 1))
        
        albef_img_cls = albef_img_cls.reshape(1,-1)
        dot_img_cls = dot_img_cls.reshape(1,-1)
        
        dict_return = {}
        dict_return['img'] = {'node_albef': albef_img_feat, # (1+576, 768)
                              'node_dot': dot_img_feat, # (1+n_obj, 768)
                              'edge_index': img_edge_index, #(2, len_edge)
                              'edge_attr': img_edge_attr, # (len_edge)
                              'cls_albef': albef_img_cls, # (1,256) 
                              'cls_dot': dot_img_cls} # (1, 768)
        
        dict_return['id'] = f"{img_id}"
        return dict_return
    
def collate_function_both(batch):
    cap_node_albef = torch.tensor(())
    cap_node_dot = torch.tensor(())
    list_cap_edge_index = []
    list_cap_edge_attr = []
    cap_cls_albef = torch.tensor(())
    cap_cls_dot = torch.tensor(())
    
    img_node_albef = torch.tensor(())
    img_node_dot = torch.tensor(())  
    list_img_edge_index = []
    list_img_edge_attr = []
    img_cls_albef = torch.tensor(())
    img_cls_dot = torch.tensor(())
    
    list_id = []
    list_n_img_node = []
    list_n_img_node_albef = []
    list_n_img_node_dot = []
    list_n_cap_node = []
    list_n_cap_node_albef = []
    list_n_cap_node_dot = []
    for x in batch:
        img_cls_albef = torch.cat((img_cls_albef, x['img']['cls_albef']), dim=0)
        img_cls_dot = torch.cat((img_cls_dot, x['img']['cls_dot']), dim=0)
        img_node_albef = torch.cat((img_node_albef, x['img']['node_albef']), dim=0)
        img_node_dot = torch.cat((img_node_dot, x['img']['node_dot']), dim=0)
        list_img_edge_index.append(x['img']['edge_index'])
        list_img_edge_attr.append(x['img']['edge_attr'])
        list_n_img_node.append(x['img']['node_albef'].shape[0] + x['img']['node_dot'].shape[0])
        list_n_img_node_albef.append(x['img']['node_albef'].shape[0])
        list_n_img_node_dot.append(x['img']['node_dot'].shape[0])
        
        cap_cls_albef = torch.cat((cap_cls_albef, x['cap']['cls_albef']), dim=0)
        cap_cls_dot = torch.cat((cap_cls_dot, x['cap']['cls_dot']), dim=0)
        cap_node_albef = torch.cat((cap_node_albef, x['cap']['node_albef']), dim=0)
        cap_node_dot = torch.cat((cap_node_dot, x['cap']['node_dot']), dim=0)
        list_cap_edge_index.append(x['cap']['edge_index'])
        list_cap_edge_attr.append(x['cap']['edge_attr'])
        list_n_cap_node.append(x['cap']['node_albef'].shape[0] + x['cap']['node_dot'].shape[0])
        list_n_cap_node_albef.append(x['cap']['node_albef'].shape[0])
        list_n_cap_node_dot.append(x['cap']['node_dot'].shape[0])
        list_id.append(x['id'])
        
    bs = len(list_id)
    img_edge_attr = torch.cat(list_img_edge_attr)
    cap_edge_attr = torch.cat(list_cap_edge_attr)
    del list_img_edge_attr, list_cap_edge_attr
    img_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_img_node))
    cap_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_cap_node))
    count_img = 0
    count_cap = 0
    for idx in range(bs):
        list_img_edge_index[idx] = list_img_edge_index[idx] + count_img
        list_cap_edge_index[idx] = list_cap_edge_index[idx] + count_cap
        count_img += list_n_img_node[idx]
        count_cap += list_n_cap_node[idx]
    img_edge_index = torch.cat(list_img_edge_index, dim=1)
    cap_edge_index = torch.cat(list_cap_edge_index, dim=1)
    del list_img_edge_index, list_cap_edge_index
    n_img_node_albef = torch.tensor(list_n_img_node_albef)
    n_img_node_dot = torch.tensor(list_n_img_node_dot)
    n_cap_node_albef = torch.tensor(list_n_cap_node_albef)
    n_cap_node_dot = torch.tensor(list_n_cap_node_dot)
    del list_n_img_node_albef, list_n_img_node_dot, list_n_cap_node_albef, list_n_cap_node_dot
    img_dict = {'cls_albef': img_cls_albef, 'cls_dot': img_cls_dot, 'batch_index': img_batch_index,
                'node_albef': img_node_albef, 'node_dot': img_node_dot,
                'n_node_albef': n_img_node_albef, 'n_node_dot': n_img_node_dot,
                'edge_index': img_edge_index, 'edge_attr': img_edge_attr}
    cap_dict = {'cls_albef': cap_cls_albef, 'cls_dot': cap_cls_dot, 'batch_index': cap_batch_index,
                'node_albef': cap_node_albef, 'node_dot': cap_node_dot,
                'n_node_albef': n_cap_node_albef, 'n_node_dot': n_cap_node_dot,
                'edge_index': cap_edge_index, 'edge_attr': cap_edge_attr}
    list_id = torch.tensor([[int(x.split('_')[0]) for x in list_id]]).reshape(-1,1)
    return img_dict, cap_dict, list_id

def collate_function_img(batch):   
    img_node_albef = torch.tensor(())
    img_node_dot = torch.tensor(())
    list_img_edge_index = []
    list_img_edge_attr = []
    img_cls_albef = torch.tensor(())
    img_cls_dot = torch.tensor(())
    list_id = []
    list_n_img_node = []
    list_n_img_node_albef = []
    list_n_img_node_dot = []
    for x in batch:
        img_cls_albef = torch.cat((img_cls_albef, x['img']['cls_albef']), dim=0)
        img_cls_dot = torch.cat((img_cls_dot, x['img']['cls_dot']), dim=0)
        img_node_albef = torch.cat((img_node_albef, x['img']['node_albef']), dim=0)
        img_node_dot = torch.cat((img_node_dot, x['img']['node_dot']), dim=0)
        list_img_edge_index.append(x['img']['edge_index'])
        list_img_edge_attr.append(x['img']['edge_attr'])
        list_n_img_node.append(x['img']['node_albef'].shape[0] + x['img']['node_dot'].shape[0])
        list_n_img_node_albef.append(x['img']['node_albef'].shape[0])
        list_n_img_node_dot.append(x['img']['node_dot'].shape[0])
        list_id.append(x['id'])
        
    bs = len(list_id)
    img_edge_attr = torch.cat(list_img_edge_attr)
    del list_img_edge_attr
    img_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_img_node))
    count_img = 0
    for idx in range(bs):
        list_img_edge_index[idx] = list_img_edge_index[idx] + count_img
        count_img += list_n_img_node[idx]
    img_edge_index = torch.cat(list_img_edge_index, dim=1)
    del list_img_edge_index
    n_img_node_albef = torch.tensor(list_n_img_node_albef)
    n_img_node_dot = torch.tensor(list_n_img_node_dot)
    img_dict = {'cls_albef': img_cls_albef, 'cls_dot': img_cls_dot, 'batch_index': img_batch_index,
                'node_albef': img_node_albef, 'node_dot': img_node_dot,
                'n_node_albef': n_img_node_albef, 'n_node_dot': n_img_node_dot,
                'edge_index': img_edge_index, 'edge_attr': img_edge_attr}
    return img_dict, list_id
  
def collate_function_cap(batch):
    cap_node_albef = torch.tensor(())
    cap_node_dot = torch.tensor(())
    list_cap_edge_index = []
    list_cap_edge_attr = []
    cap_cls_albef = torch.tensor(())
    cap_cls_dot = torch.tensor(())
    list_id = []
    list_n_cap_node = []
    list_n_cap_node_albef = []
    list_n_cap_node_dot = []
    for x in batch:
        cap_cls_albef = torch.cat((cap_cls_albef, x['cap']['cls_albef']), dim=0)
        cap_cls_dot = torch.cat((cap_cls_dot, x['cap']['cls_dot']), dim=0)
        cap_node_albef = torch.cat((cap_node_albef, x['cap']['node_albef']), dim=0)
        cap_node_dot = torch.cat((cap_node_dot, x['cap']['node_dot']), dim=0)
        list_cap_edge_index.append(x['cap']['edge_index'])
        list_cap_edge_attr.append(x['cap']['edge_attr'])
        list_n_cap_node.append(x['cap']['node_albef'].shape[0] + x['cap']['node_dot'].shape[0])
        list_n_cap_node_albef.append(x['cap']['node_albef'].shape[0])
        list_n_cap_node_dot.append(x['cap']['node_dot'].shape[0])
        list_id.append(x['id'])
        
    bs = len(list_id)
    cap_edge_attr = torch.cat(list_cap_edge_attr)
    del list_cap_edge_attr
    cap_batch_index = torch.tensor(np.repeat([x for x in range(bs)], list_n_cap_node))
    count_cap = 0
    for idx in range(bs):
        list_cap_edge_index[idx] = list_cap_edge_index[idx] + count_cap
        count_cap += list_n_cap_node[idx]
    cap_edge_index = torch.cat(list_cap_edge_index, dim=1)
    del list_cap_edge_index
    n_cap_node_albef = torch.tensor(list_n_cap_node_albef)
    n_cap_node_dot = torch.tensor(list_n_cap_node_dot)
    cap_dict = {'cls_albef': cap_cls_albef, 'cls_dot': cap_cls_dot, 'batch_index': cap_batch_index,
                'node_albef': cap_node_albef, 'node_dot': cap_node_dot,
                'n_node_albef': n_cap_node_albef, 'n_node_dot': n_cap_node_dot,
                'edge_index': cap_edge_index, 'edge_attr': cap_edge_attr}
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


class Image_Caption_Dataset_Augment(Dataset):
    def __init__(self, list_img_id_cap_id, config, para_standardize=None):        
        self.albef_dir = config['albef_dir']
        self.dot_dir = config['dot_dir']
        self.albef_augment_dir = config['albef_augment_dir']
        self.wffx = config['wffx']
        self.wfc = config['wfc']
        self.wfcx = config['wfcx']
        self.wcc = config['wcc']
        self.self_loop = True
        self.directed_graph = config['directed_graph']
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
        img_id_cap_id_split = img_id_cap_id.split('_')
        if len(img_id_cap_id_split) == 3: # include augment idx
            img_id, aug_id, cap_id = img_id_cap_id_split
        else: # normal  
            img_id, cap_id = img_id_cap_id_split
            aug_id = None
        cap_id = int(cap_id)
        # ALBEF
        if aug_id is not None:
            albef_img = np.load(f"{self.albef_augment_dir}/images/{img_id}_{aug_id}.npz")
        else:
            albef_img = np.load(f"{self.albef_dir}/images/{img_id}.npz")
        albef_cap = np.load(f"{self.albef_dir}/texts/{img_id}.npz")
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
        dot_img = np.load(f"{self.dot_dir}/images/{img_id}.npz")
        dot_cap = np.load(f"{self.dot_dir}/texts/{img_id}.npz",allow_pickle=True)
        # dot_img_feat = dot_img['feat'][1:] # the CLS is projected with additional layer (n_obj, 768)
        dot_img_feat = dot_img['feat'][0:] # the CLS is projected with additional layer (1_cls+n_obj, 768)
        dot_img_cls = dot_img['proj_embed'] # the projected CLS (768,)
        dot_cap_feat = dot_cap['feat'][cap_id] # include CLS (n_len, 768)
        dot_cap_att = dot_cap['att'][cap_id] # (n_len)
        len_cap = sum(dot_cap_att) # include the CLS
        # dot_cap_feat = dot_cap_feat[1:len_cap] # (len_cap-1, 768)
        dot_cap_feat = dot_cap_feat[0:len_cap] # (len_cap, 768) # --> 1st element is CLS (before projected as cap_cls)
        dot_cap_cls = dot_cap['proj_embed'][cap_id] # (768,)
        # Convert to tensor
        albef_cap_feat = torch.tensor(albef_cap_feat)
        albef_cap_cls = torch.tensor(albef_cap_cls)
        albef_img_feat = torch.tensor(albef_img_feat)
        albef_img_cls = torch.tensor(albef_img_cls)
        dot_cap_feat = torch.tensor(dot_cap_feat)
        dot_cap_cls = torch.tensor(dot_cap_cls)
        dot_img_feat = torch.tensor(dot_img_feat)
        dot_img_cls = torch.tensor(dot_img_cls)
        # Normalize
        if self.util_norm:
            albef_cap_feat = F.normalize(albef_cap_feat, dim=1)
            albef_cap_cls = F.normalize(albef_cap_cls)
            albef_img_feat = F.normalize(albef_img_feat, dim=1)
            albef_img_cls = F.normalize(albef_img_cls)
            dot_cap_feat = F.normalize(dot_cap_feat, dim=1)
            dot_cap_cls = F.normalize(dot_cap_cls)
            dot_img_feat = F.normalize(dot_img_feat, dim=1)
            dot_img_cls = F.normalize(dot_img_cls)
        # Normalize data
        if self.para_standardize is not None:
            if 'a_i' in list(self.para_standardize.keys()):
                albef_cap_feat = do_standardize(albef_cap_feat, self.para_standardize['a_t']['mean'].item(), self.para_standardize['a_t']['std'].item())
                # albef_cap_cls = do_standardize(albef_cap_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_img_feat = do_standardize(albef_img_feat, self.para_standardize['a_i']['mean'].item(), self.para_standardize['a_i']['std'].item())
                # albef_img_cls = do_standardize(albef_img_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                dot_cap_feat = do_standardize(dot_cap_feat, self.para_standardize['d_t']['mean'].item(), self.para_standardize['d_t']['std'].item())
                # dot_cap_cls = do_standardize(dot_cap_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_img_feat = do_standardize(dot_img_feat, self.para_standardize['d_i']['mean'].item(), self.para_standardize['d_i']['std'].item())
                # dot_img_cls = do_standardize(dot_img_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
            else:
                albef_cap_feat = do_standardize(albef_cap_feat, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_cap_cls = do_standardize(albef_cap_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_img_feat = do_standardize(albef_img_feat, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                albef_img_cls = do_standardize(albef_img_cls, self.para_standardize['a']['mean'].item(), self.para_standardize['a']['std'].item())
                dot_cap_feat = do_standardize(dot_cap_feat, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_cap_cls = do_standardize(dot_cap_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_img_feat = do_standardize(dot_img_feat, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
                dot_img_cls = do_standardize(dot_img_cls, self.para_standardize['d']['mean'].item(), self.para_standardize['d']['std'].item())
        
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
        f1_c1 = create_index_from_2_list(f1, c1, dual_index=not self.directed_graph, self_loop=self.self_loop) # (2, n_edge)
        f2_c2 = create_index_from_2_list(f2, c2, dual_index=not self.directed_graph, self_loop=self.self_loop)
        f2_c1 = create_index_from_2_list(f2, c1, dual_index=not self.directed_graph, self_loop=False)
        f1_c2 = create_index_from_2_list(f1, c2, dual_index=not self.directed_graph, self_loop=False)
        c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
        f1_f2 = create_index_from_2_list(f1, f2, dual_index=True, self_loop=False)
        edge_index = np.concatenate((f1_c1, f2_c2, c1_c2, f1_c2, f2_c1, f1_f2), axis=1)
        # create edge attr
        f1_c1_attr = [self.wfc for x in range(f1_c1.shape[1])]
        f2_c2_attr = [self.wfc for x in range(f2_c2.shape[1])]
        if self.self_loop:
            f1_c1_attr[-1] = self.wcc
            f2_c2_attr[-1] = self.wcc
        c1_c2_attr = [self.wcc for x in range(c1_c2.shape[1])]
        f1_c2_attr = [self.wfcx for x in range(f1_c2.shape[1])]
        f2_c1_attr = [self.wfcx for x in range(f2_c1.shape[1])]
        f1_f2_attr = [self.wffx for x in range(f1_f2.shape[1])]
        edge_attr = np.array(f1_c1_attr + f2_c2_attr + c1_c2_attr + f1_c2_attr + f2_c1_attr + f1_f2_attr)
        # drop ffx or not
        drop_idx = np.where(edge_attr==-1)[0]
        img_edge_attr = torch.tensor(np.delete(edge_attr, drop_idx), dtype=torch.float)
        img_edge_index = torch.tensor(np.delete(edge_index, drop_idx, 1))
        
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
        f1_c1 = create_index_from_2_list(f1, c1, dual_index=not self.directed_graph, self_loop=self.self_loop) # (2, n_edge)
        f2_c2 = create_index_from_2_list(f2, c2, dual_index=not self.directed_graph, self_loop=self.self_loop)
        f2_c1 = create_index_from_2_list(f2, c1, dual_index=not self.directed_graph, self_loop=False)
        f1_c2 = create_index_from_2_list(f1, c2, dual_index=not self.directed_graph, self_loop=False)
        c1_c2 = create_index_from_2_list(c1, c2, dual_index=True, self_loop=False)
        f1_f2 = create_index_from_2_list(f1, f2, dual_index=True, self_loop=False)
        edge_index = np.concatenate((f1_c1, f2_c2, c1_c2, f1_c2, f2_c1, f1_f2), axis=1)
        # create edge attr
        f1_c1_attr = [self.wfc for x in range(f1_c1.shape[1])]
        f2_c2_attr = [self.wfc for x in range(f2_c2.shape[1])]
        if self.self_loop:
            f1_c1_attr[-1] = self.wcc
            f2_c2_attr[-1] = self.wcc
        c1_c2_attr = [self.wcc for x in range(c1_c2.shape[1])]
        f1_c2_attr = [self.wfcx for x in range(f1_c2.shape[1])]
        f2_c1_attr = [self.wfcx for x in range(f2_c1.shape[1])]
        f1_f2_attr = [self.wffx for x in range(f1_f2.shape[1])]
        edge_attr = np.array(f1_c1_attr + f2_c2_attr + c1_c2_attr + f1_c2_attr + f2_c1_attr + f1_f2_attr)
        # drop ffx or not
        drop_idx = np.where(edge_attr==-1)[0]
        cap_edge_attr = torch.tensor(np.delete(edge_attr, drop_idx), dtype=torch.float)
        cap_edge_index = torch.tensor(np.delete(edge_index, drop_idx, 1))
        
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
        return dict_return