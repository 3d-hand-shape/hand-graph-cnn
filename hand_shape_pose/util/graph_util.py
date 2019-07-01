# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Graph utilities
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
import numpy as np
import scipy.sparse as sp
import logging

import torch

from hand_shape_pose.util.coarsening import coarsen, rescale_L, lmax_L


def load_mesh_tri(mesh_path):
    print("loading mesh triangles")

    mesh_tri = []
    with open(mesh_path) as reader:
        for line in reader:
            fields = line.strip().split()
            try:
                if fields[0] == 'f':
                    mesh_tri.append([int(f.split('/')[0]) - 1 for f in fields[1:]])
            except:
                pass
    mesh_tri = np.array(mesh_tri)
    return mesh_tri


def hand_mesh_tri(ori_mesh_tri, arm_start_index, arm_end_index):
    arm_mesh_indices = range(arm_start_index, arm_end_index)
    arm_indices_set = set(arm_mesh_indices)

    hand_mesh_tri = []

    def index_shift(ind):
        if ind >= arm_end_index:
            return ind - (arm_end_index - arm_start_index)
        else:
            return ind

    for ii in range(ori_mesh_tri.shape[0]):
        if (ori_mesh_tri[ii][0] not in arm_indices_set) and (ori_mesh_tri[ii][1] not in arm_indices_set) and (
                ori_mesh_tri[ii][2] not in arm_indices_set):
            hand_mesh_tri.append(list(map(index_shift, ori_mesh_tri[ii])))

    hand_mesh_tri = np.array(hand_mesh_tri)
    return hand_mesh_tri


def normalize_sparse_mx(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_graph(hand_tri, num_vertex):
    """
    :param hand_tri: T x 3
    :return: adj: sparse matrix, V x V (torch.sparse.FloatTensor)
    """
    num_tri = hand_tri.shape[0]
    edges = np.empty((num_tri * 3, 2))
    for i_tri in range(num_tri):
        edges[i_tri * 3] = hand_tri[i_tri, :2]
        edges[i_tri * 3 + 1] = hand_tri[i_tri, 1:]
        edges[i_tri * 3 + 2] = hand_tri[i_tri, [0, 2]]

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_vertex, num_vertex), dtype=np.float32)

    adj = adj - (adj > 1) * 1.0

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # adj = normalize_sparse_mx(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
#     if torch.cuda.is_available():
#         L = L.cuda()

    return L


def perm_index_reverse(indices):
  indices_reverse = np.copy(indices)

  for i, j in enumerate(indices):
    indices_reverse[j] = i

  return indices_reverse


def build_hand_graph(graph_template_path, output_dir):
    """
    Build graph for Hand Mesh
    """
    logger = logging.getLogger("hand_shape_pose_inference")

    hand_tri = load_mesh_tri(graph_template_path)
    arm_index_range = [473, 529]
    if len(arm_index_range) > 1 and arm_index_range[1] > arm_index_range[0]:
        hand_tri = hand_mesh_tri(hand_tri, arm_index_range[0], arm_index_range[1])

    graph_dict_path = osp.join(output_dir, 'graph_dict.npy')
    coarsening_levels = 4

    if not osp.isfile(graph_dict_path):
        # Build graph
        hand_mesh_adj = build_graph(hand_tri, hand_tri.max() + 1)
        # Compute coarsened graphs
        graph_Adj, graph_L, graph_perm = coarsen(hand_mesh_adj, coarsening_levels)

        graph_dict = {'hand_mesh_adj': hand_mesh_adj, 'coarsen_graphs_Adj': graph_Adj,
                      'coarsen_graphs_L': graph_L, 'graph_perm': graph_perm}
        np.save(graph_dict_path, graph_dict)
    else:
        logger.info("Load saved graph from {}.".format(graph_dict_path))
        graph_dict = np.load(graph_dict_path).item()
        hand_mesh_adj = graph_dict['hand_mesh_adj']
        graph_Adj = graph_dict['coarsen_graphs_Adj']
        graph_L = graph_dict['coarsen_graphs_L']
        graph_perm = graph_dict['graph_perm']

    for i, g in enumerate(graph_L):
        logger.info("Layer {0}: M_{0} = |V| = {1} nodes, |E| = {2} edges".format(i, g.shape[0], graph_Adj[i].nnz // 2))

    graph_mask = torch.from_numpy((np.array(graph_perm) < hand_tri.max() + 1).astype(float)).float()
    graph_mask = graph_mask.unsqueeze(-1).expand(-1, 3)  # V x 3

    # Compute max eigenvalue of graph Laplacians, rescale Laplacian
    graph_lmax = []
    for i in range(coarsening_levels):
        graph_lmax.append(lmax_L(graph_L[i]))
        graph_L[i] = rescale_L(graph_L[i], graph_lmax[i])

    logger.info("lmax: " + str([graph_lmax[i] for i in range(coarsening_levels)]))

    graph_perm_reverse = perm_index_reverse(graph_perm)

    return graph_L, graph_mask, graph_perm_reverse, hand_tri
