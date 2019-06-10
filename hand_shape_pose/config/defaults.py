# Copyright (c) Liuhao Ge. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_WORKERS = 0
_C.MODEL.BATCH_SIZE = 8
_C.MODEL.NUM_JOINTS = 21

_C.MODEL.HOURGLASS = CN()
_C.MODEL.HOURGLASS.NUM_STAGES = 2
_C.MODEL.HOURGLASS.NUM_MODULES = 2
_C.MODEL.HOURGLASS.NUM_FEAT_CHANNELS = 256

_C.MODEL.PRETRAIN_WEIGHT = CN()
_C.MODEL.PRETRAIN_WEIGHT.HM_NET_PATH = ""
_C.MODEL.PRETRAIN_WEIGHT.MESH_NET_PATH = ""
_C.MODEL.PRETRAIN_WEIGHT.POSE_NET_PATH = ""

_C.GRAPH = CN()
_C.GRAPH.TEMPLATE_PATH = "./data/0214_lowP_vn_g.0001.obj"

_C.EVAL = CN()
_C.EVAL.SAVE_DIR = "./output"
_C.EVAL.DATASET = ""
_C.EVAL.SAVE_BATCH_IMAGES_PRED = False
_C.EVAL.PRINT_FREQ = 5
_C.EVAL.SAVE_POSE_ESTIMATION = False
