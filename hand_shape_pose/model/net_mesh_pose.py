# Copyright (c) Liuhao Ge. All Rights Reserved.
# Some of the code for Graph ConvNet is forked from https://github.com/xbresson/spectral_graph_convnets
r"""
Networks for mesh generation and pose estimation using Spectral Graph ConvNet
"Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering",
M Defferrard, X Bresson, P Vandergheynst, NPIS 2016
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hand_shape_pose.util.net_util import FCLayer, my_sparse_mm


class Graph_CNN_Mesh_Pose(nn.Module):
    def __init__(self, num_mesh_input_chan, num_output_chan, graph_L):
        print('Graph ConvNet: mesh to pose')

        super(Graph_CNN_Mesh_Pose, self).__init__()

        self.num_mesh_input_chan = num_mesh_input_chan
        self.num_output_chan = num_output_chan
        self.graph_L = graph_L

        # parameters
        self.CL_F = [num_mesh_input_chan, 32, 64]
        self.CL_K = [3, 3]
        self.layers_per_block = [2, 2]
        self.FC_F = [self.CL_F[-1] * self.graph_L[-1].shape[0], 512]

        _cl = []
        _bn = []
        for block_i in range(len(self.CL_F) - 1):
            for layer_i in range(self.layers_per_block[block_i]):
                if layer_i == 0:
                    Fin = self.CL_K[block_i] * self.CL_F[block_i]
                else:
                    Fin = self.CL_K[block_i] * self.CL_F[block_i + 1]

                Fout = self.CL_F[block_i + 1]
                _cl.append(nn.Linear(Fin, Fout))

                scale = np.sqrt(2.0 / (Fin + Fout))
                _cl[-1].weight.data.uniform_(-scale, scale)
                _cl[-1].bias.data.fill_(0.0)

                _bn.append(nn.BatchNorm1d(Fout))

        self.cl = nn.ModuleList(_cl)
        self.bn = nn.ModuleList(_bn)

        self.fc = nn.Sequential()
        for fc_id in range(len(self.FC_F) - 1):
            self.fc.add_module('fc_%d' % (fc_id + 1),
                               FCLayer(self.FC_F[fc_id], self.FC_F[fc_id + 1], use_activation=False,
                                       use_dropout=False))

        self.fc.add_module('fc_%d' % len(self.FC_F), FCLayer(self.FC_F[-1], self.num_output_chan,
                                                             use_activation=False, use_dropout=False))

    def init_weights(self, W, Fin, Fout):
        scale = np.sqrt(2.0 / (Fin + Fout))
        W.uniform_(-scale, scale)

        return W

    def graph_conv_cheby(self, x, cl, bn, L, Fout, K):
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size()
        B, V, Fin = int(B), int(V), int(Fin)

        # transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin * B])  # V x Fin*B
        x = x0.unsqueeze(0)  # 1 x V x Fin*B

        def concat(x, x_):
            x_ = x_.unsqueeze(0)  # 1 x V x Fin*B
            return torch.cat((x, x_), 0)  # K x V x Fin*B

        if K > 1:
            x1 = my_sparse_mm()(L, x0)  # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])  # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B * V, Fin * K])  # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)  # B*V x Fout
        if bn is not None:
            x = bn(x)  # B*V x Fout
        x = x.view([B, V, Fout])  # B x V x Fout

        return x

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)  # B x F x V/p
            x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x

    def forward(self, x):
        # x: B x V x Fin
        cl_i = 0
        for block_i in range(len(self.CL_F) - 1):
            for layer_i in range(self.layers_per_block[block_i]):
                x = self.graph_conv_cheby(x, self.cl[cl_i], None, self.graph_L[block_i * 2],
                                          self.CL_F[block_i + 1], self.CL_K[block_i])
                # x = F.relu(x)
                cl_i = cl_i + 1

            x = self.graph_max_pool(x, 4)

        # FC1
        x = x.view(-1, self.FC_F[0])
        x = self.fc(x)

        return x
