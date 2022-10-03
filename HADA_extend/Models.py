from .Comp_Branch import EnLiFu
from .Comp_Basic import SeqLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class MH(nn.Module):
    def __init__(self, ft_trans=[768], ft_gcn=[768, 512], ft_com=[512, 512],
                 n_heads=4, type_gcn='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(MH, self).__init__()
        self.enc = EnLiFu(ft_trans=ft_trans, ft_gcn=ft_gcn, ft_com=ft_com, n_heads=n_heads,
                              type_graph=type_gcn, skip=skip, batch_norm=batch_norm, dropout=dropout, act_func=act_func)
    def forward(self, data):
        x = self.enc(x_albef=data['node_albef'], x_dot=data['node_dot'],
                     n_albef=data['n_node_albef'], n_dot=data['n_node_dot'],
                     edge_index=data['edge_index'], edge_attr=data['edge_attr'], 
                     batch_index=data['batch_index'], 
                     x_cls_albef=data['cls_albef'], x_cls_dot=data['cls_dot']) # (batch, ft_com[-1])
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
    
class Discriminator_2(nn.Module):
    def __init__(self, ft_in=512, dropout=0.5):
        super(Discriminator_2, self).__init__()
        self.p1 = nn.Parameter(torch.ones(ft_in))
        self.p2 = nn.Parameter(torch.ones(ft_in))
        self.do1 = nn.Dropout(p=dropout)
        self.do2 = nn.Dropout(p=dropout)
    def forward(self, feat1, feat2):
        f1n = self.do1(self.p1) * feat1
        f2n = self.do2(self.p2) * feat2
        f1n = F.normalize(f1n, dim=1)
        f2n = F.normalize(f2n, dim=1)
        dp = f1n @ f2n.T
        dpd = torch.sigmoid(dp.diag().view(-1,1))
        return dpd