from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.utils import *
from hand_shape_pose.util.vis import *


def init_pose3d_labels(cam_param_path, pose3d_gt_path):
    all_camera_params = load_camera_param(cam_param_path)
    all_global_pose3d_gt = load_global_pose3d_gt(pose3d_gt_path)
    return all_camera_params, all_global_pose3d_gt


def read_data(im_path, all_camera_params, all_global_pose3d_gt, global_mesh_gt_dir):
    """
    read the corresponding pose and mesh ground truth of the image sample, and camera parameters
    :param im_path:
    :param all_camera_params: (N_pose, N_cam, 7) focal_length, 3 translation val; 3 euler angles (degree)
    :param all_global_pose3d_gt: (N_pose, 21, 3)
    :param global_mesh_gt_dir:
    :return:
    """
    pose_id, camera_id = extract_pose_camera_id(osp.basename(im_path))

    cam_param = all_camera_params[pose_id][camera_id]  # (7, )

    # get ground truth of 3D hand pose
    global_pose3d_gt = all_global_pose3d_gt[pose_id]  # (21, 3)
    local_pose3d_gt = transform_global_to_cam(global_pose3d_gt, cam_param)  # (21, 3)

    # get ground truth of 3D hand mesh
    mesh_files = glob.glob(osp.join(global_mesh_gt_dir, "*.%04d.obj" % (pose_id + 1)))
    assert len(mesh_files) == 1, "Cannot find a unique mesh file for pose %04d" % (pose_id + 1)
    mesh_file = mesh_files[0]
    global_mesh_pts_gt, global_mesh_normal_gt, mesh_tri_idx = load_mesh_from_obj(mesh_file)
    # global_mesh_pts_gt: (N_vertex, 3), global_mesh_normal_gt: (N_tris, 3)
    # mesh_tri_idx: (N_tris, 3)

    local_mesh_pts_gt = transform_global_to_cam(global_mesh_pts_gt, cam_param)  # (N_vertex, 3)
    local_mesh_normal_gt = transform_global_to_cam(global_mesh_normal_gt, cam_param)

    return local_pose3d_gt, local_mesh_pts_gt, local_mesh_normal_gt, cam_param, mesh_tri_idx


def visualize_data(im_path, local_pose3d_gt, local_mesh_pts_gt, cam_param, mesh_tri_idx):
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    img_rgb = img[:, :, :3]
    img_mask = img[:, :, 3:]
    img_wo_bkg = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask)
    # for training data, you can use the hand mask to blend hand image with random background images

    im_height, im_width = img.shape[:2]
    fig = plt.figure()
    fig.set_size_inches(float(4 * im_height) / fig.dpi, float(4 * im_width) / fig.dpi, forward=True)

    # 1. plot raw image
    ax = plt.subplot(231)
    ax.imshow(img_rgb)
    ax.set_title("raw image")

    # 2. plot image without background
    ax = plt.subplot(232)
    ax.imshow(img_wo_bkg)
    ax.set_title("image without background")

    # 3. plot 2D joints
    ax = plt.subplot(233)

    fl = cam_param[0]  # focal length
    cam_proj_mat = np.array([[fl,  0.0, im_width / 2.],
                             [0.0, fl,  im_height / 2.],
                             [0.0, 0.0, 1.0]])
    pose_2d = cam_projection(local_pose3d_gt, cam_proj_mat)

    skeleton_overlay = draw_2d_skeleton(img_rgb, pose_2d)
    ax.imshow(skeleton_overlay)
    ax.set_title("image with GT 2D joints")

    # 4. plot 3D joints
    ax = plt.subplot(234, projection='3d')
    draw_3d_skeleton_on_ax(local_pose3d_gt, ax)
    ax.set_title("GT 3D joints")

    # 5. plot 3D mesh
    ax = plt.subplot(235, projection='3d')
    ax.plot_trisurf(local_mesh_pts_gt[:, 0], local_mesh_pts_gt[:, 1], local_mesh_pts_gt[:, 2],
                    triangles=mesh_tri_idx, color='grey', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-85, azim=-75)
    ax.set_title("GT 3D mesh")

    # 6. plot 2D mesh points
    ax = plt.subplot(236)
    mesh_2d = cam_projection(local_mesh_pts_gt, cam_proj_mat)
    ax.imshow(img_rgb)
    ax.scatter(mesh_2d[:, 0], mesh_2d[:, 1], s=15, color='green', alpha=0.8)
    ax.set_title("image with GT mesh 2D projection points")

    ret = fig2data(fig)
    plt.close(fig)

    cv2.imwrite('./data/example_synthetic.jpg', ret)


def main():
    parser = argparse.ArgumentParser(description="View Synthetic 3D Hand Shape and Pose Dataset")
    parser.add_argument(
        "--image-path",
        help="path to image file",
    )
    parser.add_argument(
        "--camera-param-path",
        default="./data/synthetic_train_val/3D_labels/camPosition.txt",
        help="path to file of camera parameters",
    )
    parser.add_argument(
        "--global-pose3d-gt-path",
        default="./data/synthetic_train_val/3D_labels/handGestures.txt",
        help="path to file of global 3D hand pose ground truth",
    )
    parser.add_argument(
        "--global-mesh-gt-dir",
        default="./data/synthetic_train_val/hand_3D_mesh",
        help="path to global 3D hand mesh ground truth directory",
    )

    args = parser.parse_args()

    all_camera_params, all_global_pose3d_gt = \
        init_pose3d_labels(args.camera_param_path, args.global_pose3d_gt_path)

    local_pose3d_gt, local_mesh_pts_gt, local_mesh_normal_gt, cam_param, mesh_tri_idx = \
        read_data(args.image_path, all_camera_params, all_global_pose3d_gt, args.global_mesh_gt_dir)

    visualize_data(args.image_path, local_pose3d_gt, local_mesh_pts_gt, cam_param, mesh_tri_idx)


if __name__ == "__main__":
    main()
