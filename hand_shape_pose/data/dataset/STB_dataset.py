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
import numpy.linalg as LA
import math

import torch
import torch.utils.data

from hand_shape_pose.util.image_util import crop_pad_im_from_bounding_rect

BB_base = 120.054 / 10.0  # cm
BB_fx = 822.79041
BB_fy = 822.79041
BB_tx = 318.47345
BB_ty = 250.31296

SK_fx_color = 607.92271
SK_fy_color = 607.88192
SK_tx_color = 314.78337
SK_ty_color = 236.42484


def SK_rot_mx(rot_vec):
    """
    use Rodrigues' rotation formula to transform the rotation vector into rotation matrix
    :param rot_vec:
    :return:
    """
    theta = LA.norm(rot_vec)
    vector = np.array(rot_vec) * math.sin(theta / 2.0) / theta
    a = math.cos(theta / 2.0)
    b = -vector[0]
    c = -vector[1]
    d = -vector[2]
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)],
                     [2 * (b * c - a * d), a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
                     [2 * (b * d + a * c), 2 * (c * d - a * b), a * a + d * d - b * b - c * c]])


SK_rot_vec = [0.00531, -0.01196, 0.00301]
SK_trans_vec = [-24.0381, -0.4563, -1.2326]  # mm
SK_rot = SK_rot_mx(SK_rot_vec)

STB_joints = ['loc_bn_palm_L', 'loc_bn_pinky_L_01', 'loc_bn_pinky_L_02', 'loc_bn_pinky_L_03',
              'loc_bn_pinky_L_04', 'loc_bn_ring_L_01', 'loc_bn_ring_L_02', 'loc_bn_ring_L_03',
              'loc_bn_ring_L_04', 'loc_bn_mid_L_01', 'loc_bn_mid_L_02', 'loc_bn_mid_L_03',
              'loc_bn_mid_L_04', 'loc_bn_index_L_01', 'loc_bn_index_L_02', 'loc_bn_index_L_03',
              'loc_bn_index_L_04', 'loc_bn_thumb_L_01', 'loc_bn_thumb_L_02', 'loc_bn_thumb_L_03',
              'loc_bn_thumb_L_04'
              ]
snap_joint_names = ['loc_bn_palm_L', 'loc_bn_thumb_L_01', 'loc_bn_thumb_L_02', 'loc_bn_thumb_L_03',
                    'loc_bn_thumb_L_04', 'loc_bn_index_L_01', 'loc_bn_index_L_02', 'loc_bn_index_L_03',
                    'loc_bn_index_L_04', 'loc_bn_mid_L_01', 'loc_bn_mid_L_02', 'loc_bn_mid_L_03',
                    'loc_bn_mid_L_04', 'loc_bn_ring_L_01', 'loc_bn_ring_L_02', 'loc_bn_ring_L_03',
                    'loc_bn_ring_L_04', 'loc_bn_pinky_L_01', 'loc_bn_pinky_L_02', 'loc_bn_pinky_L_03',
                    'loc_bn_pinky_L_04'
                    ]
snap_joint_name2id = {w: i for i, w in enumerate(snap_joint_names)}
STB_joint_name2id = {w: i for i, w in enumerate(STB_joints)}
STB_to_Snap_id = [snap_joint_name2id[joint_name] for joint_name in STB_joints]
STB_ori_dim = [480, 640]
resize_dim = [256, 256]


class STBDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir_list, image_prefix, bbox_file, ann_file_list):
        self.image_paths = []
        self.bboxes = []
        self.pose_roots = []
        self.pose_scales = []
        self.pose_gts = []
        self.cam_params = torch.tensor([SK_fx_color, SK_fy_color, SK_tx_color, SK_ty_color])

        root_id = snap_joint_name2id['loc_bn_palm_L']

        for image_dir, ann_file in zip(image_dir_list, ann_file_list):
            mat_gt = sio.loadmat(ann_file)
            curr_pose_gts = mat_gt["handPara"].transpose((2, 1, 0))  # N x K x 3
            curr_pose_gts = self.SK_xyz_depth2color(curr_pose_gts, SK_trans_vec, SK_rot)
            curr_pose_gts = curr_pose_gts[:, STB_to_Snap_id, :] / 10.0  # convert to Snap index, mm->cm
            curr_pose_gts = self.palm2wrist(curr_pose_gts)  # N x K x 3
            curr_pose_gts = torch.from_numpy(curr_pose_gts)
            self.pose_gts.append(curr_pose_gts)

            self.pose_roots.append(curr_pose_gts[:, root_id, :])  # N x 3
            self.pose_scales.append(self.compute_hand_scale(curr_pose_gts))  # N

            for image_id in range(curr_pose_gts.shape[0]):
                self.image_paths.append(osp.join(image_dir, "%s_%d.png" % (image_prefix, image_id)))

        self.pose_roots = torch.cat(self.pose_roots, 0).float()
        self.pose_scales = torch.cat(self.pose_scales, 0).float()
        self.pose_gts = torch.cat(self.pose_gts, 0).float()

        mat_bboxes = sio.loadmat(bbox_file)
        self.bboxes = torch.from_numpy(mat_bboxes["bboxes"]).float()  # N x 4

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        '''crop image'''
        crop_img = crop_pad_im_from_bounding_rect(img, self.bboxes[index, :].int())
        '''resize image'''
        crop_resized_img = cv2.resize(crop_img, (resize_dim[1], resize_dim[0]))
        crop_resized_img = torch.from_numpy(crop_resized_img)  # 256 x 256 x 3

        return crop_resized_img, self.cam_params, self.bboxes[index], \
               self.pose_roots[index], self.pose_scales[index], index

    def __len__(self):
        return len(self.image_paths)

    def SK_xyz_depth2color(self, depth_xyz, trans_vec, rot_mx):
        """
        :param depth_xyz: N x 21 x 3, trans_vec: 3, rot_mx: 3 x 3
        :return: color_xyz: N x 21 x 3
        """
        color_xyz = depth_xyz - np.tile(trans_vec, [depth_xyz.shape[0], depth_xyz.shape[1], 1])
        return color_xyz.dot(rot_mx)

    def palm2wrist(self, pose_xyz):
        root_id = snap_joint_name2id['loc_bn_palm_L']
        ring_root_id = snap_joint_name2id['loc_bn_ring_L_01']
        pose_xyz[:, root_id, :] = pose_xyz[:, ring_root_id, :] + \
                                  2.0 * (pose_xyz[:, root_id, :] - pose_xyz[:, ring_root_id, :])  # N x K x 3
        return pose_xyz

    def compute_hand_scale(self, pose_xyz):
        ref_bone_joint_1_id = snap_joint_name2id['loc_bn_mid_L_02']
        ref_bone_joint_2_id = snap_joint_name2id['loc_bn_mid_L_01']

        pose_scale_vec = pose_xyz[:, ref_bone_joint_1_id, :] - pose_xyz[:, ref_bone_joint_2_id, :]  # N x 3
        pose_scale = torch.norm(pose_scale_vec, dim=1)  # N
        return pose_scale

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
