from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.utils import *
from hand_shape_pose.util.vis import *


def init_3d_labels(param_file_path, pose3d_gt_path):
    mat_params = sio.loadmat(param_file_path)
    image_paths = mat_params["image_path"]
    cam_params = mat_params["cam_param"]  # N x 4, [fx, fy, u0, v0]
    bboxes = mat_params["bbox"]  # N x 4, bounding box in the original image, [x, y, w, h]

    cropped_cam_params = []
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(osp.join(osp.dirname(param_file_path), image_path))
        bbox = bboxes[i]
        cam_param = cam_params[i]
        resize_ratio = float(img.shape[0]) / bbox[2]
        cropped_cam_param = np.array([cam_param[0], cam_param[1], cam_param[2] - bbox[0],
                                      cam_param[3] - bbox[1]]) * resize_ratio
        cropped_cam_params.append(cropped_cam_param)
    cropped_cam_params = np.stack(cropped_cam_params)

    mat_gt = sio.loadmat(pose3d_gt_path)
    all_pose3d_gt = mat_gt["pose_gt"]  # N x 21 x 3

    return cropped_cam_params, all_pose3d_gt


def read_data(im_path, all_camera_params, all_pose3d_gt, mesh_gt_dir):
    """

    :param im_path:
    :param all_camera_params: (N, 4) [fx, fy, u0, v0]
    :param all_pose3d_gt: (N, 21, 3)
    :param mesh_gt_dir:
    :return:
    """
    frame_id = int(osp.splitext(osp.basename(im_path))[0])
    pose3d_gt = all_pose3d_gt[frame_id]
    cam_param = all_camera_params[frame_id]  # (4, )

    # get ground truth of 3D hand mesh
    mesh_file = osp.join(mesh_gt_dir, "%05d.obj" % frame_id)
    mesh_pts_gt, mesh_normal_gt, mesh_tri_idx = load_mesh_from_obj(mesh_file)
    # mesh_pts_gt: (N_vertex, 3), mesh_normal_gt: (N_tris, 3), mesh_tri_idx: (N_tris, 3)

    return pose3d_gt, mesh_pts_gt, mesh_normal_gt, cam_param, mesh_tri_idx


def visualize_data(im_path, local_pose3d_gt, local_mesh_pts_gt, cam_param, mesh_tri_idx):
    img = cv2.imread(im_path)

    im_height, im_width = img.shape[:2]
    fig = plt.figure()
    fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

    # 1. plot raw image
    ax = plt.subplot(231)
    ax.imshow(img)
    ax.set_title("real-world image")

    # 2. plot 2D joints
    ax = plt.subplot(232)
    cam_proj_mat = np.array([[cam_param[0], 0.0,          cam_param[2]],
                             [0.0,          cam_param[1], cam_param[3]],
                             [0.0,          0.0,          1.0]])
    pose_2d = cam_projection(local_pose3d_gt, cam_proj_mat)
    skeleton_overlay = draw_2d_skeleton(img, pose_2d)
    ax.imshow(skeleton_overlay)
    ax.set_title("image with GT 2D joints")

    # 3. plot 3D joints
    ax = plt.subplot(234, projection='3d')
    draw_3d_skeleton_on_ax(local_pose3d_gt, ax)
    ax.set_title("GT 3D joints")

    # 4. plot 3D mesh
    ax = plt.subplot(235, projection='3d')
    ax.plot_trisurf(local_mesh_pts_gt[:, 0], local_mesh_pts_gt[:, 1], local_mesh_pts_gt[:, 2],
                    triangles=mesh_tri_idx, color='grey', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-85, azim=-75)
    ax.set_title("GT 3D mesh")

    # 5. plot 2D mesh points
    ax = plt.subplot(236)
    mesh_2d = cam_projection(local_mesh_pts_gt, cam_proj_mat)
    ax.imshow(img)
    ax.scatter(mesh_2d[:, 0], mesh_2d[:, 1], s=15, color='green', alpha=0.8)
    ax.set_title("image with GT mesh 2D projection points")

    ret = fig2data(fig)
    plt.close(fig)

    cv2.imwrite('./data/example_realworld.jpg', ret)


def main():
    parser = argparse.ArgumentParser(description="View Real-world 3D Hand Shape and Pose Dataset")
    parser.add_argument(
        "--image-path",
        help="path to image file",
    )
    parser.add_argument(
        "--param-file-path",
        default="./data/real_world_testset/params.mat",
        help="path to file of camera parameters",
    )
    parser.add_argument(
        "--pose3d-gt-path",
        default="./data/real_world_testset/pose_gt.mat",
        help="path to file of 3D hand pose ground truth",
    )
    parser.add_argument(
        "--mesh-gt-dir",
        default="./data/real_world_testset/real_hand_3D_mesh",
        help="path to global 3D hand mesh ground truth directory",
    )

    args = parser.parse_args()

    cam_params, pose_gts = \
        init_3d_labels(args.param_file_path, args.pose3d_gt_path)

    pose3d_gt, mesh_pts_gt, mesh_normal_gt, cam_param, mesh_tri_idx = \
        read_data(args.image_path, cam_params, pose_gts, args.mesh_gt_dir)

    visualize_data(args.image_path, pose3d_gt, mesh_pts_gt, cam_param, mesh_tri_idx)


if __name__ == "__main__":
    main()
