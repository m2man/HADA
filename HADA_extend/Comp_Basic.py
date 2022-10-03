import torch
import torch.nn.functional as F
import torch.nn as nn
from .Utils import MemoryEfficientSwish
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm
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

def select_graph_layer(type_model='GCN'):
    if type_model == 'GCN':
        return GCNConv
    if type_model == 'GATv2':
        return GATv2Conv
    if type_model == 'TGCN':
        return TransformerConv

class DWSConv1D(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer=1, kernel_depth=3, padding_depth=1):
        super(DWSConv1D, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin * kernels_per_layer, kernel_size=kernel_depth, padding=padding_depth, groups=nin)
        self.pointwise = nn.Conv1d(nin * kernels_per_layer, nout, kernel_size=1)
        nn.init.kaiming_normal_(self.depthwise.weight)
        nn.init.kaiming_normal_(self.pointwise.weight)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
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
            if idx != len(ft_out)-1:
                if batch_norm:
                    # self.norm.append(nn.BatchNorm1d(ft_out[idx]))
                    self.norm.append(nn.LayerNorm([ft_out[idx]]))
                else:
                    self.norm.append(nn.Identity())
                self.act.append(get_activate_func(act_func))
            self.dropout.append(nn.Dropout(p=dropout))
            
        self.linear = nn.ModuleList(self.linear)
        for x in self.linear:
            nn.init.kaiming_normal_(x.weight)
        self.norm = nn.ModuleList(self.norm)
        self.dropout = nn.ModuleList(self.dropout)
        self.act = nn.ModuleList(self.act)
        
    def forward(self, x):
        # x.shape (in_channel, ft_in)
        for idx in range(len(self.linear)):
            # x = self.linear[idx](x) 
            # if idx != (len(self.linear)-1): # last layer not use relu
            #     x = self.norm[idx](x)
            #     x = self.act[idx](x)
            # x = self.dropout[idx](x)
            # C10
            x = self.dropout[idx](x)
            x = self.linear[idx](x)
            if idx != (len(self.linear)-1): # last layer not use relu
                x = self.act[idx](x)
                x = self.norm[idx](x)
        return x  
    
    
class GraphLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels=[32], type_model='GCN', n_heads=4, 
                 skip=False, concat=False, dropout=0.5, batch_norm=True, act_func='relu'):
        super().__init__()
        assert type_model in ['GCN', 'GATv2', 'TGCN']
        self.use_batch_norm = batch_norm
        self.type_model = type_model
        self.conv = []
        self.conv_lin = []
        self.norm = []
        self.act = []
        self.p = dropout
        self.concat = concat
        self.n_heads = n_heads
        if len(hidden_channels) > 1 and skip:
            self.skip = skip
        else:
            self.skip = False
        # self.dropout = []
        for idx in range(len(hidden_channels)):
            graph_layer = select_graph_layer(self.type_model)
            if idx == 0:
                if self.type_model in ['GATv2', 'TGCN']:
                    if self.concat:
                        self.conv.append(graph_layer(in_channels, hidden_channels[idx], heads=n_heads, 
                                                     concat=self.concat, edge_dim=1, dropout=self.p))
                        self.conv_lin.append(SeqLinear(self.n_heads*hidden_channels[idx], 
                                                       [hidden_channels[idx]], 
                                                       dropout, batch_norm, act_func))
                    else:
                        self.conv.append(graph_layer(in_channels, hidden_channels[idx], heads=n_heads, 
                                                     concat=self.concat, edge_dim=1, dropout=self.p))
                else:
                    self.conv.append(graph_layer(in_channels, hidden_channels[idx], 
                                                 add_self_loops=False))
            else:
                if self.type_model in ['GATv2', 'TGCN']:
                    self.conv.append(graph_layer(hidden_channels[idx-1], hidden_channels[idx], heads=n_heads, 
                                                               concat=self.concat, edge_dim=1, dropout=self.p))
                    self.conv_lin.append(SeqLinear(self.n_heads*hidden_channels[idx], 
                                                   [hidden_channels[idx]], 
                                                   dropout, batch_norm, act_func))
                else:
                    self.conv.append(graph_layer(hidden_channels[idx-1], hidden_channels[idx], 
                                                add_self_loops=False))
            if idx != len(hidden_channels)-1:
                if batch_norm:
                    # self.norm.append(nn.BatchNorm1d(hidden_channels[idx]))
                    self.norm.append(LayerNorm(hidden_channels[idx]))
                else:
                    self.norm.append(nn.Identity())
                self.act.append(get_activate_func(act_func))
                # self.dropout.append(nn.Dropout(p=dropout))
            
        self.conv = nn.ModuleList(self.conv)
        self.norm = nn.ModuleList(self.norm)
        # self.dropout = nn.ModuleList(self.dropout)
        self.act = nn.ModuleList(self.act)
        self.conv_lin = nn.ModuleList(self.conv_lin)
        
    def forward(self, x, edge_index, edge_attr=None, batch_index=None):
        # x.shape (num_nodes in a batch, ft)
        # edge_index (2, total edges)
        # batch (num_nodes in a batch, )
        if self.type_model in ['GATv2', 'TGCN'] and edge_attr is not None:
            edge_attr = edge_attr.view(-1,1)
        for idx in range(len(self.conv)):
            if edge_attr is not None:
                xout = self.conv[idx](x, edge_index, edge_attr)
            else:
                xout = self.conv[idx](x, edge_index)
            if self.concat and self.type_model in ['GATv2', 'TGCN']:
                xout = self.conv_lin[idx](xout)
            if self.skip:
                x = x + xout
            else:
                x = xout
            if idx != (len(self.conv)-1):
                x = self.act[idx](x)
                x = self.norm[idx](x, batch_index)
            # x = self.dropout[idx](x)
            # edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, training=self.training)
            
        return x, edge_index, edge_attr # (num_nodes in a batch, hidden_channel)  