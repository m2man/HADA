import torch
import torch.nn.functional as F
import torch.nn as nn
from .Utils import MemoryEfficientSwish
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import dropout_adj

def get_activate_func(act_func=None):
    if act_func is None or act_func.lower() == 'id':
        return nn.Identity()
    if act_func.lower() == 'relu':
        return nn.ReLU()
    if act_func.lower() == 'swish':
        return MemoryEfficientSwish()
    if act_func.lower() == 'tanh':
        return nn.Tanh()
    if act_func.lower() == 'gelu':
        return nn.GELU()
    if act_func.lower() == 'elu':
        return nn.ELU()
    
class SeqLinear(nn.Module):
    def __init__(self, ft_in, ft_out=[128], dropout=0.5, batch_norm=True, act_func='relu'):
        super(SeqLinear, self).__init__()
        self.linear = []
        self.norm = []
        self.dropout = []
        self.act = []
        for idx in range(len(ft_out)):
            if idx == 0:
                self.linear.append(nn.Linear(ft_in, ft_out[idx]))
            else:
                self.linear.append(nn.Linear(ft_out[idx-1], ft_out[idx]))
            if batch_norm:
                # self.norm.append(nn.BatchNorm1d(ft_out[idx]))
                self.norm.append(nn.LayerNorm([ft_out[idx]]))
            else:
                self.norm.append(nn.Identity())
            self.dropout.append(nn.Dropout(p=dropout))
            self.act.append(get_activate_func(act_func))
            
        self.linear = nn.ModuleList(self.linear)
        for x in self.linear:
            nn.init.kaiming_normal_(x.weight)
        self.norm = nn.ModuleList(self.norm)
        self.dropout = nn.ModuleList(self.dropout)
        self.act = nn.ModuleList(self.act)
        
    def forward(self, x):
        # x.shape (in_channel, ft_in)
        for idx in range(len(self.norm)):
            x = self.linear[idx](x)
            if idx != (len(self.linear)-1): # last layer not use relu
                x = self.norm[idx](x)
                x = self.act[idx](x)
            x = self.dropout[idx](x)
        return x  
    
class Discriminator(nn.Module):
    def __init__(self, ft_in=512, ft_out=[128,1], dropout=0.5, batch_norm=True, act_func='relu'):
        super(Discriminator, self).__init__()
        self.disc = SeqLinear(ft_in=ft_in*4, ft_out=ft_out,
                             batch_norm=batch_norm, dropout=dropout, act_func=act_func)
    def forward(self, feat1, feat2):
        dist = torch.abs(feat1-feat2)
        mul = torch.mul(feat1, feat2)
        return torch.sigmoid(self.disc(torch.cat([feat1, feat2, dist, mul], dim=1)))

class MH(nn.Module):
    def __init__(self, ft_in, ft_out=[128], batch_norm=True, dropout=0.5, act_func='relu'):
        super(MH, self).__init__()
        self.enc = SeqLinear(ft_in=ft_in, ft_out=ft_out, dropout=dropout, batch_norm=batch_norm, act_func=act_func)
    def forward(self, data):
        x_cls_albef = data['cls_albef']
        x_cls_dot = data['cls_dot']
        x_in = torch.cat((x_cls_albef, x_cls_dot), dim=1)
        x = self.enc(x_in)
        return x
