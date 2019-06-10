# Copyright (c) Liuhao Ge. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "data"
    DATASETS = {
        "real_world_testset": {
            "root_dir": "real_world_testset",
            "param_file": "real_world_testset/params.mat",
            "ann_file": "real_world_testset/pose_gt.mat",
        },
        "STB_eval": {
            "root_dir": "STB",
            "image_list": ["B1Counting", "B1Random"],
            "image_prefix": "SK_color",
            "bbox_file": "STB_eval_bboxes.mat",
            "ann_dir": "STB/labels",
        }
    }

    @staticmethod
    def get(name):
        if name == "real_world_testset":
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root_dir"]),
                param_file=os.path.join(data_dir, attrs["param_file"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="RealWorldTestSet",
                args=args,
            )
        elif "STB" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            root_dir = os.path.join(data_dir, attrs["root_dir"])
            ann_dir = os.path.join(data_dir, attrs["ann_dir"])
            args = dict(
                image_dir_list=[os.path.join(root_dir, image_dir) for image_dir in attrs["image_list"]],
                image_prefix=attrs["image_prefix"],
                bbox_file=os.path.join(data_dir, attrs["bbox_file"]),
                ann_file_list=[os.path.join(ann_dir, image_dir+"_"+attrs["image_prefix"][:2]+".mat")
                               for image_dir in attrs["image_list"]],
            )
            return dict(
                factory="STBDataset",
                args=args,
            )
