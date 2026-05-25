# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:53:24 2023

@author: ab978
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GATv2Conv
from torch_geometric_temporal.nn.recurrent import DCRNN,A3TGCN
import sys, pdb


class RecurrentGCN(torch.nn.Module):
    def __init__(self, 
                 num_features, 
                 hidden_channels, 
                 output_node_channels, 
                 dropout=0.0,
                 activation='relu'):
        super(RecurrentGCN, self).__init__()
        self.num_features = num_features
        self.activation = activation
        self.dropout = dropout
        self.grnn = A3TGCN(in_channels=1, 
                            out_channels=hidden_channels,
                            periods=12)
        self.linear = torch.nn.Linear(hidden_channels, output_node_channels)
        # Apply Xavier initialization to the linear layers
        torch.nn.init.xavier_uniform_(self.linear.weight)
        
        if self.activation == 'relu':
            self.a = F.relu
        elif self.activation == 'tanh':
            self.a = F.tanh
        elif self.activation == 'sigmoid':
            self.a = F.sigmoid        
        else:
            print('Choose an available activation function')
            sys.exit()
        

   
        
    def forward(self, data):

        x, edge_index, edge_attr = (data.x,
                                       data.edge_index,
                                       data.edge_attr)
        # pdb.set_trace()
        h = self.grnn(x, edge_index, edge_attr)
        h = self.a(h)
        if self.dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)
        y_x = self.linear(h)

        return y_x.view(-1)

class GATModel(torch.nn.Module):
    def __init__(self, 
                 num_node_features,
                 num_edge_features,
                 num_heads, 
                 output_node_channels, 
                 dim_hidden_layers=[100], 
                 dropout_att=0.0, 
                 dropout=0.0,
                 activation='relu',
                 concat_heads=False,
                 negative_slope=0.2,
                 standardize = False):
        super(GATModel, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.standardize = standardize
        

        if self.standardize:
            self.bnorm_node = nn.BatchNorm1d(num_node_features, affine=False)
            self.bnorm_edge = nn.BatchNorm1d(num_edge_features, affine=False)

        
        
        if len(dim_hidden_layers) == 1:
            first_gat = [GATConv(in_channels=num_node_features, out_channels=dim_hidden_layers[0], heads=num_heads,  
                                        concat=False, dropout=dropout_att, edge_dim=num_edge_features, negative_slope=negative_slope)]
        else:
            first_gat = [GATConv(in_channels=num_node_features, out_channels=dim_hidden_layers[0], heads=num_heads,  
                                        concat=concat_heads, dropout=dropout_att, edge_dim=num_edge_features, negative_slope=negative_slope)]
                     
        stacked_gats = []
        
        for i in range(len(dim_hidden_layers)-1):
            if i+1 == len(dim_hidden_layers)-1:
                if concat_heads and num_heads>1:
                    stacked_gats.append(GATConv(in_channels=dim_hidden_layers[i] * num_heads, 
                                                out_channels=dim_hidden_layers[i+1], heads=num_heads, 
                                                concat=False, dropout=dropout_att, edge_dim=num_edge_features, negative_slope=negative_slope))
                else:
                    stacked_gats.append(GATConv(in_channels=dim_hidden_layers[i], 
                                                out_channels=dim_hidden_layers[i+1], heads=num_heads, 
                                                concat=False, dropout=dropout_att, edge_dim=num_edge_features, negative_slope=negative_slope))
            else:
                if concat_heads and num_heads>1:
                    stacked_gats.append(GATConv(in_channels=dim_hidden_layers[i] * num_heads, 
                                                out_channels=dim_hidden_layers[i+1], heads=num_heads, 
                                                concat=concat_heads, dropout=dropout_att, edge_dim=num_edge_features, negative_slope=negative_slope))
                else:
                    stacked_gats.append(GATConv(in_channels=dim_hidden_layers[i], 
                                                out_channels=dim_hidden_layers[i+1], heads=num_heads, 
                                                concat=concat_heads, dropout=dropout_att, edge_dim=num_edge_features, negative_slope=negative_slope))

        

        self.gat_layers = nn.ModuleList(first_gat + stacked_gats)


        self.linear = torch.nn.Linear(dim_hidden_layers[-1], output_node_channels)
        # Apply Xavier initialization to the linear layers
        # torch.nn.init.xavier_uniform_(self.linear.weight)

        
        if self.activation == 'relu':
            self.a = F.relu
        elif self.activation == 'tanh':
            self.a = F.tanh
        elif self.activation == 'sigmoid':
            self.a = F.sigmoid        
        else:
            print('Choose an available activation function')
            sys.exit()



    def forward(self, data):
        x, edge_index, edge_attr = (data.x,
                                    data.edge_index,
                                    data.edge_attr)

        if self.standardize:
            x = self.bnorm_node(x)
            edge_attr = self.bnorm_edge(edge_attr)
        for l in self.gat_layers:
            x = l(x, edge_index, edge_attr)
            x = self.a(x)
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)

        return x.view(-1) 