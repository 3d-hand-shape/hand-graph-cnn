# Copyright (c) Liuhao Ge. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def draw_mesh(mesh_renderer, image, cam_param, box, mesh_xyz):
    """
    :param mesh_renderer:
    :param image: H x W x 3
    :param cam_param: fx, fy, u0, v0
    :param box: x, y, w, h
    :param mesh_xyz: M x 3
    :return:
    """
    resize_ratio = float(image.shape[0]) / box[2]
    cam_for_render = np.array([cam_param[0], cam_param[2] - box[0], cam_param[3] - box[1]]) * resize_ratio

    rend_img_overlay = mesh_renderer(mesh_xyz, cam=cam_for_render, img=image, do_alpha=True)
    vps = [60.0, -60.0]
    rend_img_vps = [mesh_renderer.rotated(mesh_xyz, vp, cam=cam_for_render, img_size=image.shape[:2]) for vp in vps]

    return rend_img_overlay, rend_img_vps[0], rend_img_vps[1]


def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = np.copy(image)

    marker_sz = 6
    line_wd = 3
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(
                skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(
                skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay


def draw_3d_skeleton_on_ax(pose_cam_xyz, ax):
    """
    :param pose_cam_xyz: 21 x 3
    :param ax:
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-85, azim=-75)


def draw_3d_skeleton(pose_cam_xyz, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)
    ax = plt.subplot(111, projection='3d')
    draw_3d_skeleton_on_ax(pose_cam_xyz, ax)

    ret = fig2data(fig)  # H x W x 4
    plt.close(fig)
    return ret


def save_batch_image_with_mesh_joints(mesh_renderer, batch_images, cam_params, bboxes,
                                      est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz,
                                      file_name, padding=2):
    """
    :param mesh_renderer:
    :param batch_images: B x H x W x 3 (torch.Tensor)
    :param cam_params: B x 4 (torch.Tensor)
    :param bboxes: B x 4 (torch.Tensor)
    :param est_mesh_cam_xyz: B x 1280 x 3 (torch.Tensor)
    :param est_pose_uv: B x 21 x 2 (torch.Tensor)
    :param est_pose_cam_xyz: B x 21 x 3 (torch.Tensor)
    :param file_name:
    :param padding:
    :return:
    """
    num_images = batch_images.shape[0]
    image_height = batch_images.shape[1]
    image_width = batch_images.shape[2]
    num_column = 6

    grid_image = np.zeros((num_images * (image_height + padding), num_column * (image_width + padding), 3),
                          dtype=np.uint8)

    for id_image in range(num_images):
        image = batch_images[id_image].numpy()
        cam_param = cam_params[id_image].numpy()
        box = bboxes[id_image].numpy()
        mesh_xyz = est_mesh_cam_xyz[id_image].numpy()
        pose_uv = est_pose_uv[id_image].numpy()
        pose_xyz = est_pose_cam_xyz[id_image].numpy()

        rend_img_overlay, rend_img_vp1, rend_img_vp2 = draw_mesh(mesh_renderer, image, cam_param, box, mesh_xyz)
        skeleton_overlay = draw_2d_skeleton(image, pose_uv)
        skeleton_3d = draw_3d_skeleton(pose_xyz, image.shape[:2])

        img_list = [image, rend_img_overlay, rend_img_vp1, rend_img_vp2, skeleton_overlay, skeleton_3d]

        height_begin = (image_height + padding) * id_image
        height_end = height_begin + image_height
        width_begin = 0
        width_end = image_width
        for show_img in img_list:
            grid_image[height_begin:height_end, width_begin:width_end, :] = show_img[..., :3]
            width_begin += (image_width + padding)
            width_end = width_begin + image_width

    cv2.imwrite(file_name, grid_image)
