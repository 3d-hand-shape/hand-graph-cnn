# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Real world test set
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import scipy.io as sio
import os.path as osp
import logging
import cv2
import numpy as np

import torch
import torch.utils.data


class RealWorldTestSet(torch.utils.data.Dataset):
    def __init__(self, root, param_file, ann_file):
        self.data_path = root

        mat_params = sio.loadmat(param_file)
        self.image_paths = mat_params["image_path"]

        self.cam_params = torch.from_numpy(mat_params["cam_param"]).float()  # N x 4, [fx, fy, u0, v0]
        assert len(self.image_paths) == self.cam_params.shape[0]

        self.bboxes = torch.from_numpy(mat_params["bbox"]).float()  # N x 4, bounding box in the original image, [x, y, w, h]
        assert len(self.image_paths) == self.bboxes.shape[0]

        self.pose_roots = torch.from_numpy(mat_params["pose_root"]).float()  # N x 3, [root_x, root_y, root_z]
        assert len(self.image_paths) == self.pose_roots.shape[0]

        if "pose_scale" in mat_params.keys():
            self.pose_scales = torch.from_numpy(mat_params["pose_scale"]).squeeze().float()  # N, length of first bone of middle finger
        else:
            self.pose_scales = torch.ones(len(self.image_paths)) * 5.0
        assert len(self.image_paths) == self.pose_scales.shape[0]

        mat_gt = sio.loadmat(ann_file)
        self.pose_gts = torch.from_numpy(mat_gt["pose_gt"])  # N x K x 3
        assert len(self.image_paths) == self.pose_gts.shape[0]

    def __getitem__(self, index):
        img = cv2.imread(osp.join(self.data_path, self.image_paths[index]))
        img = torch.from_numpy(img)  # 256 x 256 x 3

        return img, self.cam_params[index], self.bboxes[index], \
               self.pose_roots[index], self.pose_scales[index], index

    def __len__(self):
        return len(self.image_paths)

    def evaluate_pose(self, results_pose_cam_xyz, save_results=False, output_dir=""):
        avg_est_error = 0.0
        for image_id, est_pose_cam_xyz in results_pose_cam_xyz.items():
            dist = est_pose_cam_xyz - self.pose_gts[image_id]  # K x 3
            avg_est_error += dist.pow(2).sum(-1).sqrt().mean()

        avg_est_error /= len(results_pose_cam_xyz)

        if save_results:
            eval_results = {}
            image_ids = results_pose_cam_xyz.keys()
            image_ids.sort()
            eval_results["image_ids"] = np.array(image_ids)
            eval_results["gt_pose_xyz"] = [self.pose_gts[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["est_pose_xyz"] = [results_pose_cam_xyz[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["gt_pose_xyz"] = torch.cat(eval_results["gt_pose_xyz"], 0).numpy()
            eval_results["est_pose_xyz"] = torch.cat(eval_results["est_pose_xyz"], 0).numpy()
            sio.savemat(osp.join(output_dir, "pose_estimations.mat"), eval_results)

        return avg_est_error.item()
