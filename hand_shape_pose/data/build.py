# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Build dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hand_shape_pose.config.paths_catalog import DatasetCatalog
from . import dataset as D


def build_dataset(dataset_name):
    data = DatasetCatalog.get(dataset_name)
    args = data["args"]

    # make dataset from factory
    factory = getattr(D, data["factory"])
    dataset = factory(**args)

    return dataset
