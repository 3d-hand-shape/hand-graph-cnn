from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
import numpy as np
import math
import glob


def get_train_val_im_paths(image_dir, val_set_path, train_val_flag):
    """
    get training or validation image paths
    :param image_dir:
    :param val_set_path:
    :param train_val_flag:
    :return:
    """
    val_cameras = []
    with open(val_set_path) as reader:
        for line in reader:
            val_cameras.append(line.strip())
    val_cameras = set(val_cameras)

    lighting_folders = glob.glob(osp.join(image_dir, "l*"))

    image_paths = []
    for lighting_folder in lighting_folders:
        cam_folders = glob.glob(osp.join(lighting_folder, "cam*"))
        for cam_folder in cam_folders:
            cam_name = osp.basename(cam_folder)
            is_val = (cam_name in val_cameras)
            if (train_val_flag == 'val' and is_val) or \
                    (train_val_flag == 'train' and not is_val):
                image_paths += glob.glob(osp.join(cam_folder, "*.png"))

    return image_paths


def extract_pose_camera_id(im_filename):
    """
    extract pose id and camera id from image file name
    :param im_filename: e.g., 'handV2_rgt01_specTest5_gPoses_ren_25cRrRs_l21_cam01_.0001.png'
    :return: pose id (int, start from 0) and camera id (int, start from 0)
    """
    name = osp.splitext(im_filename)[0]
    fields = name.split('_')
    pose_id = int(fields[-1].replace('.', '0')) - 1
    camera_id = int(fields[-2][3:]) - 1
    return pose_id, camera_id


def load_camera_param(camera_param_path):
    """
    load camera parameters
    :param camera_param_path:
    :return: (N_pose, N_cam, 7) (focal_length, 3 translation val; 3 euler angles)
    """
    all_camera_names = np.loadtxt(camera_param_path, usecols=(0,), dtype=np.str)
    num_cameras = len(np.unique(all_camera_names))
    all_camera_params = np.loadtxt(camera_param_path, usecols=(1, 2, 3, 4, 5, 6, 7))
    all_camera_params = all_camera_params.reshape((-1, num_cameras, 7))
    return all_camera_params


def load_global_pose3d_gt(pose3d_gt_path):
    """
    load global 3D hand pose ground truth
    :param pose3d_gt_path:
    :return: (N_pose, 21, 3)
    """
    all_joint_names = np.loadtxt(pose3d_gt_path, usecols=(0,), dtype=np.str)
    num_joints = len(np.unique(all_joint_names))
    all_global_pose3d_gt = np.loadtxt(pose3d_gt_path, usecols=(1, 2, 3)).reshape((-1, num_joints, 3))
    return all_global_pose3d_gt


def euler_xyz_to_rot_mx(euler_angle):
    """
    convert xyz euler angles to rotation matrix
    :param euler_angle: euler angles for x, y, z axis, (degree)
    :return: rotation matrix, (3, 3)
    """
    rad = euler_angle * math.pi / 180.0
    sins = np.sin(rad)
    coss = np.cos(rad)
    rot_x = np.array([[1, 0, 0],
                      [0, coss[0], -sins[0]],
                      [0, sins[0], coss[0]]])
    rot_y = np.array([[coss[1], 0, sins[1]],
                      [0, 1, 0],
                      [-sins[1], 0, coss[1]]])
    rot_z = np.array([[coss[2], -sins[2], 0],
                      [sins[2], coss[2], 0],
                      [0, 0, 1]])
    rot_mx = rot_z.dot(rot_y).dot(rot_x)
    return rot_mx


def transform_global_to_cam(global_3d, camera_param, use_translation=True):
    """
    transform 3D pose in global coordinate system to camera coordinate system
    :param global_3d: (N, 3)
    :param camera_param: (7, ) focal_length, 3 translation val; 3 euler angles (degree)
    :param use_translation: bool
    :return: camera_3d: (N, 3)
    """
    if use_translation:
        translation = camera_param[1:4]  # (3, )
        pose3d = global_3d - translation
    else:
        pose3d = global_3d

    theta = camera_param[4:]  # (3, )
    rot_mx = euler_xyz_to_rot_mx(theta)
    aux_mx = np.eye(3, dtype=np.float)
    aux_mx[1, 1] = -1.0
    aux_mx[2, 2] = -1.0
    rot_mx = rot_mx.dot(aux_mx)

    camera_3d = pose3d.dot(rot_mx)
    return camera_3d


def cam_projection(local_pose3d, cam_proj_mat):
    """
    get 2D projection points
    :param local_pose3d: (N, 3)
    :param cam_proj_mat: (3, 3)
    :return:
    """
    xyz = local_pose3d.dot(cam_proj_mat.transpose())  # (N, 3)
    z_inv = 1.0 / xyz[:, 2]  # (N, ), 1/z
    z_inv = np.expand_dims(z_inv, axis=1)  # (N, 1), 1/z
    xyz = xyz * z_inv
    pose_2d = xyz[:, :2]  # (N, 2)
    return pose_2d


def load_mesh_from_obj(mesh_file, arm_index_range=[473, 529]):
    """
    Load mesh vertices, normals, triangle indices and vertices from obj file
    :param mesh_file: path to the hand mesh obj file
    :param arm_index_range: range of indices which belong to arm
    :return: mesh vertices, normals, triangle indices and vertices
    """
    mesh_pts = []
    mesh_tri_idx = []
    mesh_vn = []
    id_vn = 0
    state = 'V'
    with open(mesh_file) as reader:
        for line in reader:
            fields = line.strip().split()
            try:
                if fields[0] == 'v':
                    if state != 'V':
                        break
                    mesh_pts.append([float(f) for f in fields[1:]])
                if fields[0] == 'f':
                    state = 'F'
                    mesh_tri_idx.append([int(f.split('/')[0]) - 1 for f in fields[1:]])
                if fields[0] == 'vn':
                    state = 'N'
                    if id_vn % 3 == 0:
                        mesh_vn.append([float(f) for f in fields[1:]])
                    id_vn = id_vn + 1
            except:
                pass

    mesh_pts = np.array(mesh_pts)  # (N_vertex, 3)
    mesh_vn = np.array(mesh_vn)  # (N_tris, 3)
    mesh_tri_idx = np.array(mesh_tri_idx)  # (N_tris, 3)

    if len(arm_index_range) > 1 and arm_index_range[1] > arm_index_range[0]:
        mesh_pts, mesh_vn, mesh_tri_idx = \
            remove_arm_vertices(mesh_pts, mesh_vn, mesh_tri_idx, arm_index_range)

    return mesh_pts, mesh_vn, mesh_tri_idx


def get_mesh_tri_vertices(mesh_vertices, mesh_tri_idx):
    """
    get the 3D coordinates of three vertices in mesh triangles
    :param mesh_vertices: (N_vertex, 3)
    :param mesh_tri_idx: (N_tris, 3)
    :return: (N_tris, 3, 3)
    """
    mesh_tri_pts = np.zeros((len(mesh_tri_idx), 3, 3))  # (N_tris, 3, 3)
    for idx, tri in enumerate(mesh_tri_idx):
        mesh_tri_pts[idx, 0, :] = mesh_vertices[tri[0]]
        mesh_tri_pts[idx, 1, :] = mesh_vertices[tri[1]]
        mesh_tri_pts[idx, 2, :] = mesh_vertices[tri[2]]

    return mesh_tri_pts


def remove_arm_vertices(mesh_pts, mesh_vn, mesh_tri_idx, arm_index_range):
    """
    remove vertices belong to arm in the hand mesh
    :param mesh_pts: (N_vertex, 3)
    :param mesh_vn: (N_tris, 3)
    :param mesh_tri_idx: (N_tris, 3)
    :param arm_index_range: range of indices which belong to arm
    :return:
    """
    arm_mesh_idx = range(arm_index_range[0], arm_index_range[1])
    arm_index_set = set(arm_mesh_idx)
    hand_indices = list(set(range(0, len(mesh_pts))) - arm_index_set)
    hand_mesh_pts = mesh_pts[hand_indices]

    hand_mesh_tri_idx = []
    hand_mesh_vn = []

    if mesh_tri_idx.size <= 1:
        return hand_mesh_pts, hand_mesh_vn, hand_mesh_tri_idx

    def _index_shift(ind):
        if ind >= arm_index_range[1]:
            return ind - (arm_index_range[1] - arm_index_range[0])
        else:
            return ind

    for i in range(mesh_tri_idx.shape[0]):
        if (mesh_tri_idx[i][0] not in arm_index_set) and (mesh_tri_idx[i][1] not in arm_index_set) and \
                (mesh_tri_idx[i][2] not in arm_index_set):
            hand_mesh_tri_idx.append(list(map(_index_shift, mesh_tri_idx[i])))
            hand_mesh_vn.append(mesh_vn[i])

    hand_mesh_tri_idx = np.array(hand_mesh_tri_idx)
    hand_mesh_vn = np.array(hand_mesh_vn)
    return hand_mesh_pts, hand_mesh_vn, hand_mesh_tri_idx
