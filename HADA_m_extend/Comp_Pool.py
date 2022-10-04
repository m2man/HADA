from torch_geometric.nn import DenseGraphConv, DMoNPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse, dropout_adj
import torch
import torch.nn as nn
from .Comp_Basic import *

class PoGaDW(nn.Module):
    def __init__(self, in_channels, out_channels = [20], feature_size=[32], batch_norm=True, dropout=0.5,
                 kernels_per_layer=1, kernel_depth=3, padding_depth='same', act_func='relu'):
        super(PoGaDW, self).__init__()
        self.conv = []
        self.norm = []
        self.pool = []
        self.dropout = []
        self.act = []
        for idx in range(len(feature_size)):
            if idx == 0:
                self.conv.append(DWSConv1D(nin=in_channels, nout=out_channels[idx], 
                                           kernels_per_layer=kernels_per_layer, kernel_depth=kernel_depth,
                                           padding_depth=padding_depth))
            else:
                self.conv.append(DWSConv1D(nin=out_channels[idx-1], nout=out_channels[idx],
                                           kernels_per_layer=kernels_per_layer, kernel_depth=kernel_depth,
                                           padding_depth=padding_depth))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(out_channels[idx]))
            else:
                self.norm.append(nn.Identity())
            self.pool.append(nn.AdaptiveMaxPool1d(feature_size[idx]))
            self.dropout.append(nn.Dropout(p=dropout))
            self.act.append(get_activate_func(act_func))
            
        self.pool = nn.ModuleList(self.pool)
        self.conv = nn.ModuleList(self.conv)
        self.norm = nn.ModuleList(self.norm)
        self.dropout = nn.ModuleList(self.dropout)
        self.act = nn.ModuleList(self.act)
        
    def forward(self, x):
        # x.shape (batch, in_channels, feature_size)
        for idx in range(len(self.pool)):
            x = self.conv[idx](x)
            x = self.norm[idx](x)
            if idx != (len(self.pool)-1): # last layer not use relu
                x = self.act[idx](x)
            x = self.dropout[idx](x)
            x = self.pool[idx](x)
        return x  
    
class PoGaCNN(nn.Module):
    def __init__(self, in_channels, out_channels = [20], feature_size=[32], kernel_size=[3], batch_norm=True, dropout=0.5, act_func='relu'):
        super(PoGaCNN, self).__init__()
        self.conv = []
        self.norm = []
        self.dropout = []
        self.act = []
        self.pool = []
        for idx in range(len(out_channels)): # reduce number of channels (nodes) -- keep feature same size
            if idx == 0:
                self.conv.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels[idx], 
                                           kernel_size=kernel_size[idx], padding='same'))
            else:
                self.conv.append(nn.Conv1d(in_channels=out_channels[idx-1], out_channels=out_channels[idx], 
                                           kernel_size=kernel_size[idx], padding='same'))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(out_channels[idx]))
            else:
                self.norm.append(nn.Identity())
            self.dropout.append(nn.Dropout(p=dropout))
            self.act.append(get_activate_func(act_func))
            self.pool.append(nn.AdaptiveMaxPool1d(feature_size[idx]))
            
        self.conv = nn.ModuleList(self.conv)
        self.norm = nn.ModuleList(self.norm)
        self.dropout = nn.ModuleList(self.dropout)
        self.pool = nn.ModuleList(self.pool)
        self.act = nn.ModuleList(self.act)
        
    def forward(self, x):
        # x.shape (batch, in_channels, feature_size)
        for idx in range(len(self.pool)):
            x = self.conv[idx](x)
            x = self.norm[idx](x)
            if idx != (len(self.pool)-1): # last layer not use relu
                x = self.act[idx](x)
            x = self.dropout[idx](x)
            x = self.pool[idx](x)
        return x  

def un_batch(batch_x, batch_adj, device='cpu'):
    bs, nnode, hs = batch_x.shape
    temp = torch.zeros(nnode*bs, nnode*bs)
    for x in range(bs):
        temp[(x*nnode):((x+1)*nnode), (x*nnode):((x+1)*nnode)] = batch_adj[x]
    new_adj_0, new_adj_1 = torch.where(temp!=0)
    new_edge_index = torch.concat((new_adj_0, new_adj_1)).view(2,-1)
    new_edge_index = new_edge_index.to(device)
    new_batch = torch.tensor([x for x in range(bs)])
    new_batch = new_batch.repeat_interleave(nnode)
    new_batch = new_batch.to(device)
    new_x = batch_x.view([-1,hs])
    return new_x, new_edge_index, new_batch

class PoGa(nn.Module):
    def __init__(self, in_channels, num_nodes = [20], hidden_channels=[32], type_model='GCN', dropout=0.1, act_func='relu', device='cuda'):
        super().__init__()
        assert type_model in ['GCN', 'GATv2', 'TGCN']
        self.type_model = type_model
        self.device = device
        self.pool = []
        self.conv = []
        self.act = []
        for idx in range(len(num_nodes)):
            graph_layer = select_graph_layer(self.type_model)
            if idx == 0:
                self.conv.append(graph_layer(in_channels, hidden_channels[idx]))
            else:
                if self.type_model[:3] == 'GCN':
                    self.conv.append(DenseGraphConv(hidden_channels[idx-1], hidden_channels[idx]))
                else:
                    self.conv.append(graph_layer(hidden_channels[idx-1], hidden_channels[idx]))
            self.pool.append(DMoNPooling([hidden_channels[idx], hidden_channels[idx]], num_nodes[idx], dropout=dropout))
            self.act.append(get_activate_func(act_func))
        
        if self.type_model[:3] == 'GCN':
            self.conv.append(DenseGraphConv(hidden_channels[-1], hidden_channels[-1]))
        else:
            self.conv.append(graph_layer(hidden_channels[-1], hidden_channels[-1]))      
        
        self.pool = nn.ModuleList(self.pool)
        self.conv = nn.ModuleList(self.conv)
        
    def forward(self, x, edge_index, batch):
        # x.shape (num_nodes in a batch, ft)
        # edge_index (2, total edges)
        # batch (num_nodes in a batch, )
        sp_loss = 0
        o_loss = 0
        c_loss = 0
        for idx in range(len(self.pool)):
            x = self.conv[idx](x, edge_index)
            x = self.act[idx](x)
            if idx == 0 or self.type_model != 'GCN':
                x, mask = to_dense_batch(x, batch) # x shape batch, max nodes, ft
                edge_index = to_dense_adj(edge_index, batch)
            else:
                mask = None
            _, x, edge_index, sp, o, c = self.pool[idx](x, edge_index, mask)
            if self.type_model != 'GCN':
                x, edge_index, batch = un_batch(x, edge_index, self.device)
            sp_loss += sp
            o_loss += o
            c_loss += c
        x = self.conv[-1](x, edge_index)
        if self.type_model != 'GCN':
            x, mask = to_dense_batch(x, batch)
        # x.shape (new_num_nodes in a batch, new_ft)
        # new_num_nodes now can be equally between graphs
        pool_loss = sp_loss + o_loss + c_loss
        return x, pool_loss 