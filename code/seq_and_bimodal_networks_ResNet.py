import torch
from torch import nn
import torch.nn.functional as F
import sys
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor


class GraphAttention(nn.Module):
    def __init__(self,
                 F,
                 F_, 
                 attn_heads=1, 
                 attn_heads_reduction='concat',
                 dropout_rate=0.0, 
                 activation='relu', 
                 use_bias=False,
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, 
                 bias_regularizer=None,
                 attn_kernel_regularizer=None, 
                 activity_regularizer=None,
                 kernel_constraint=None, 
                 bias_constraint=None,
                 attn_kernel_constraint=None, 
                 **kwargs):
        super(GraphAttention, self).__init__()

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possible reduction methods: concat, average')

        self.F = F
        self.F_ = F_
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "elu":
            self.activation = torch.nn.ELU()
        self.use_bias = use_bias
        
        if kernel_initializer=='glorot_uniform':
            self.kernel_initializer = torch.nn.init.xavier_uniform
        if bias_initializer=='zeros':
            self.bias_initializer = torch.nn.init.zeros_
        if attn_kernel_initializer=='glorot_uniform':
            self.attn_kernel_initializer = torch.nn.init.xavier_uniform

        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.attn_kernel_regularizer = None
        self.activity_regularizer = None

        self.kernel_constraint = None
        self.bias_constraint = None
        self.attn_kernel_constraint = None

        self.supports_masking = False
        
        self.kernels = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.attn_kernels = nn.ParameterList()
        
        if attn_heads_reduction == 'concat':
            self.output_dim = self.F_ * self.attn_heads
        else:
            self.output_dim = self.F_

        self.build()

    def build(self):

        for head in range(self.attn_heads):
            kernel_self = nn.Parameter(self.kernel_initializer(torch.empty(self.F, self.F_)))
            kernel_neighs = nn.Parameter(self.kernel_initializer(torch.empty(self.F, self.F_)))
            self.kernels.append(nn.ParameterList([kernel_self, kernel_neighs]))
    
            if self.use_bias:
                bias = nn.Parameter(self.bias_initializer(torch.empty(self.F_)))
                self.biases.append(bias)
            
            attn_kernel_self = nn.Parameter(self.attn_kernel_initializer(torch.empty(self.F_, 1)))
            attn_kernel_neighs = nn.Parameter(self.attn_kernel_initializer(torch.empty(self.F_, 1)))
            self.attn_kernels.append(nn.ParameterList([attn_kernel_self, attn_kernel_neighs]))
    
    def forward(self, X, A):
        outputs = []
        Att = []
        for head in range(self.attn_heads):
            kernel_self = self.kernels[head][0]
            kernel_neighs = self.kernels[head][1]
            attention_kernel = self.attn_kernels[head]
            
            features_self = torch.matmul(X, kernel_self)
            features_neighs = torch.matmul(X, kernel_neighs)

            attn_for_self = torch.matmul(features_self, attention_kernel[0])
            attn_for_neighs = torch.matmul(features_neighs, attention_kernel[1])

            attn_for_self_permute = attn_for_self.permute(1, 0, 2)
            attn_for_neighs_permute = attn_for_neighs.permute(1, 0, 2)
            att = attn_for_self_permute + torch.transpose(attn_for_neighs_permute, 0, 2)
            att = att.permute(1, 0, 2)

            att = F.leaky_relu(att, negative_slope=0.2)
            mask = -10e15 * (1.0 - A)
            att += mask

            att = torch.sigmoid(att)
            att_sum = torch.sum(att, dim=-1, keepdim=True)
            att = att / (1 + att_sum)
            beta_promoter = 1 / (1 + att_sum)

            Att.append(att)

            dropout_feat_neigh = F.dropout(features_neighs, self.dropout_rate)
            dropout_feat_self = F.dropout(features_self, self.dropout_rate)

            node_features = dropout_feat_self * beta_promoter + torch.bmm(att, dropout_feat_neigh)

            if self.use_bias:
                node_features = node_features + self.biases[head]

            outputs.append(node_features)

        if self.attn_heads_reduction == 'concat':
            output = torch.cat(outputs, dim=-1)
        else:
            output = torch.mean(torch.stack(outputs), dim=0)
        output = self.activation(output)
        
        return output, Att



class permute(nn.Module):
    def forward(self,x):
        return torch.permute(x, (0, 2, 1))



class MyExpLayer(nn.Module):
    def __init__(self):
        super(MyExpLayer, self).__init__()

    def forward(self, x):
        return torch.exp(x)

class MyLinearLayer(nn.Module):
    def __init__(self):
        super(MyLinearLayer, self).__init__()

    def forward(self, x):
        return x
        

class seq_network(nn.Module):
    def __init__(self, model_params_dict, chromatin_tracks_path_list, num_layers=34, image_channels=4, num_classes=1, l2_reg=0.01, dropout_rate=0.5):
        super(seq_network, self).__init__()

        self.dropout_rate = model_params_dict["dropout"]
        
        self.conv1 = nn.Conv1d(image_channels, 32, kernel_size=21, padding='same')
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.maxpool1 = nn.MaxPool1d(2)

        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(2)

        self.dropout3 = nn.Dropout(p=self.dropout_rate)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.relu3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.maxpool3 = nn.MaxPool1d(5)

        self.dropout4 = nn.Dropout(p=self.dropout_rate)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding='same')
        self.relu4 = nn.ReLU()
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.maxpool4 = nn.MaxPool1d(5)

        self.dropout_dilation_list = nn.ModuleList()
        self.conv_dilation_list = nn.ModuleList()
        self.relu_dilation_list = nn.ModuleList()
        self.bn_dilation_list = nn.ModuleList()
        for i in range(1,1+6):
            dropout_dilation = nn.Dropout(p=self.dropout_rate)
            conv_dilation = nn.Conv1d(64, 64, kernel_size=3, dilation=2**i, padding='same')
            relu_dilation = nn.ReLU()
            bn_dilation = nn.BatchNorm1d(64)
            self.dropout_dilation_list.append(dropout_dilation)
            self.conv_dilation_list.append(conv_dilation)
            self.relu_dilation_list.append(relu_dilation)
            self.bn_dilation_list.append(bn_dilation)
         
        self.chromatin_track_conv_list=nn.ModuleList()
        self.chromatin_track_activation_list=nn.ModuleList()
        for i in range(len(chromatin_tracks_path_list)):
            conv = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding='same')
            
            activation = nn.ReLU()
            
            self.chromatin_track_conv_list.append(conv)
            self.chromatin_track_activation_list.append(activation)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)
        x = self.maxpool1(x)

        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)
        x = self.maxpool2(x)

        x = self.dropout3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batch_norm3(x)
        x = self.maxpool3(x)
        
        x = self.dropout4(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.batch_norm4(x)
        x = self.maxpool4(x)
 
        print(x.shape)

        for dropout, conv, relu, bn in zip(self.dropout_dilation_list,
                                           self.conv_dilation_list,
                                           self.relu_dilation_list,
                                           self.bn_dilation_list):
            x = dropout(x)
            x = relu(conv(x)) + x
            x = bn(x)

        print(x.shape)
        
        out_list =[]
        for layer, activation in zip(self.chromatin_track_conv_list,self.chromatin_track_activation_list):
            out = layer(x)
            out = activation(out)
            out = torch.squeeze(out)
            out_list.append(out)
        
        print(out_list[0].shape)
        
        return out_list


class GAT_network(nn.Module):
    def __init__(self, model_params_dict):
        super().__init__()
        
        self.nnodes = model_params_dict["nnodes"]
        self.out_features = model_params_dict["out_features"]
        self.n_gat_layers = 4
        self.attn_heads = 4
        self.att = nn.ParameterList()
        self.bn_layers = nn.ParameterList()
        
        gat_layer = GraphAttention(F=128,
                                   F_=self.out_features, 
                                   attn_heads=self.attn_heads, 
                                   dropout_rate=0.5,
                                   attn_heads_reduction='concat',
                                   activation="elu"
                                  )
        bn_layer = nn.BatchNorm1d(self.nnodes)
        self.att.append(gat_layer)
        self.bn_layers.append(bn_layer)
        for i in range(self.n_gat_layers):
            gat_layer = GraphAttention(F=self.attn_heads*self.out_features,
                                       F_=self.out_features,
                                       attn_heads=self.attn_heads, 
                                       dropout_rate=0.5,
                                       attn_heads_reduction='concat',
                                       activation="elu"
                                      )
            bn_layer = nn.BatchNorm1d(self.nnodes)
            self.att.append(gat_layer)
            self.bn_layers.append(bn_layer)
            
        self.permute = permute()
        
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, padding='same')
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(2)
 
        self.dropout2 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding='same')
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.maxpool2 = nn.MaxPool1d(5)
        
        self.dropout3 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(32, 1, kernel_size=3, padding='same')
        self.batch_norm3 = nn.BatchNorm1d(1)
        self.maxpool3 = nn.MaxPool1d(2)
        
#         self.dropout4 = nn.Dropout(0.5)
#         self.conv4 = nn.Conv1d(32, 16, kernel_size=1, padding='same')
#         self.batch_norm4 = nn.BatchNorm1d(16)
#         self.maxpool4 = nn.MaxPool1d(5)
        
#         self.dropout5 = nn.Dropout(0.5)
#         self.conv5 = nn.Conv1d(16, 8, kernel_size=1, padding='same')
#         self.batch_norm5 = nn.BatchNorm1d(8)
#         self.maxpool5 = nn.MaxPool1d(2)
        
#         self.dropout6 = nn.Dropout(0.5)
#         self.conv6 = nn.Conv1d(8, 8, kernel_size=1, padding='same')
#         self.batch_norm6 = nn.BatchNorm1d(8)
#         self.maxpool6 = nn.MaxPool1d(5)
        
#         self.dropout7 = nn.Dropout(0.5)
#         self.conv7 = nn.Conv1d(8, 1, kernel_size=1, padding='same')
#         self.batch_norm7 = nn.BatchNorm1d(1)
#         self.maxpool7 = nn.MaxPool1d(2)
       
    def forward(self,X,adj):
        att=[]
        x=X
        x = self.permute(x)
        print("#"*20)
        print(x.shape)
        for gat_layer, bn_layer in zip(self.att, self.bn_layers):
            x, att_ = gat_layer(x, adj)
            x = bn_layer(x)
            att.append(att_)

        x = self.permute(x)
        print(f"GAT output shape = {x.shape}")
        
        x = nn.functional.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.maxpool1(x)
        
        x = self.dropout2(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.maxpool2(x)
        
        x = self.dropout3(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.maxpool3(x)
        
#         x = self.dropout4(x)
#         x = nn.functional.relu(self.conv4(x))
#         x = self.batch_norm4(x)
#         x = self.maxpool4(x)
  
#         x = self.dropout5(x)
#         x = nn.functional.relu(self.conv5(x))
#         x = self.batch_norm5(x)
#         x = self.maxpool5(x)
        
#         x = self.dropout6(x)
#         x = nn.functional.relu(self.conv6(x))
#         x = self.batch_norm6(x)
#         x = self.maxpool6(x)

#         x = self.dropout7(x)
#         x = nn.functional.relu(self.conv7(x))
#         x = self.batch_norm7(x)
#         x = self.maxpool7(x)

        print(x.shape)
        
        x = x.reshape(-1,1)
        
        print(f"GAT output shape = {x.shape}")
        
        return x


class bimodal_network_GAT(nn.Module):
    def __init__(self, model_params_dict, chromtracks_path, base_model):
        super().__init__()
        
        print("init - bimodal_network_GAT")
        
        self.nnodes = model_params_dict["nnodes"]
        self.out_features = model_params_dict["out_features"]

        # https://stackoverflow.com/questions/52796121/how-to-get-the-output-from-a-specific-layer-from-a-pytorch-model
        return_nodes = {
            "maxpool4": "maxpool4",
        }
        self.base_model = create_feature_extractor(base_model, return_nodes=return_nodes)

        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding='same')
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(2)

        self.chrom_model=GAT_network(model_params_dict)
        self.sigmoid=nn.Sigmoid()

    def forward(self, seq_input, adj):
        
        print("  Forward - bimodal_network_GAT")
        
        intermediate_outputs = self.base_model(seq_input)
        h = intermediate_outputs["maxpool4"]
 
        print(h.shape)
        x = nn.functional.relu(self.conv1(h))
        x = self.batch_norm1(x)
        x = self.maxpool1(x)

        x = nn.functional.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.maxpool2(x)
        
        print(x.shape)
        print(adj.shape)

        xc=self.chrom_model(x, adj)

        print(f"  h = {h.shape}")
        print(f"  adj = {adj.shape}")
        print(f"  xc = {xc.shape}")

        result=self.sigmoid(xc)
        
        return result
