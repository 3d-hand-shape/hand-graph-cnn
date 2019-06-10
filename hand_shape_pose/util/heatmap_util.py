# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Utilities for heat-map
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


def find_keypoints_max(heatmaps):
  """
  heatmaps: C x H x W
  return: C x 3
  """
  # flatten the last axis
  heatmaps_flat = heatmaps.view(heatmaps.size(0), -1)

  # max loc
  max_val, max_ind = heatmaps_flat.max(1)
  max_ind = max_ind.float()

  max_v = torch.floor(torch.div(max_ind, heatmaps.size(1)))
  max_u = torch.fmod(max_ind, heatmaps.size(2))
  return torch.cat((max_u.view(-1,1), max_v.view(-1,1), max_val.view(-1,1)), 1)

def compute_uv_from_heatmaps(hm, resize_dim):
  """
  :param hm: B x K x H x W (Variable)
  :param resize_dim:
  :return: uv in resize_dim (Variable)
  """
  upsample = nn.Upsample(size=resize_dim, mode='bilinear')  # (B x K) x H x W
  resized_hm = upsample(hm).view(-1, resize_dim[0], resize_dim[1])

  uv_confidence = find_keypoints_max(resized_hm)  # (B x K) x 3

  return uv_confidence.view(-1, hm.size(1), 3)
