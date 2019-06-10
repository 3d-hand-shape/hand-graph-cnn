# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Basic evaluation script for PyTorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path as osp
import torch

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import save_batch_image_with_mesh_joints
from hand_shape_pose.util import renderer


def main():
    parser = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = osp.join(cfg.EVAL.SAVE_DIR, args.config_file)
    mkdir(output_dir)
    logger = setup_logger("hand_shape_pose_inference", output_dir, filename='eval-' + get_logger_filename())
    logger.info(cfg)

    # 1. Load network model
    model = ShapePoseNetwork(cfg, output_dir)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.load_model(cfg)

    mesh_renderer = renderer.MeshRenderer(model.hand_tri.astype('uint32'))

    # 2. Load data
    dataset_val = build_dataset(cfg.EVAL.DATASET)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.MODEL.BATCH_SIZE,
        num_workers=cfg.MODEL.NUM_WORKERS
    )

    # 3. Inference
    model.eval()
    results_pose_cam_xyz = {}
    cpu_device = torch.device("cpu")
    logger.info("Evaluate on {} frames:".format(len(dataset_val)))
    for i, batch in enumerate(data_loader_val):
        images, cam_params, bboxes, pose_roots, pose_scales, image_ids = batch
        images, cam_params, bboxes, pose_roots, pose_scales = \
            images.to(device), cam_params.to(device), bboxes.to(device), pose_roots.to(device), pose_scales.to(device)
        with torch.no_grad():
            est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz = \
                model(images, cam_params, bboxes, pose_roots, pose_scales)

            est_mesh_cam_xyz = [o.to(cpu_device) for o in est_mesh_cam_xyz]
            est_pose_uv = [o.to(cpu_device) for o in est_pose_uv]
            est_pose_cam_xyz = [o.to(cpu_device) for o in est_pose_cam_xyz]

        results_pose_cam_xyz.update({img_id.item(): result for img_id, result in zip(image_ids, est_pose_cam_xyz)})

        if i % cfg.EVAL.PRINT_FREQ == 0:
            # 4. evaluate pose estimation
            avg_est_error = dataset_val.evaluate_pose(results_pose_cam_xyz, save_results=False)  # cm
            msg = 'Evaluate: [{0}/{1}]\t' 'Average pose estimation error: {2:.2f} (mm)'.format(
                len(results_pose_cam_xyz), len(dataset_val), avg_est_error * 10.0)
            logger.info(msg)

            # 5. visualize mesh and pose estimation
            if cfg.EVAL.SAVE_BATCH_IMAGES_PRED:
                file_name = '{}_{}.jpg'.format(osp.join(output_dir, 'pred'), i)
                logger.info("Saving image: {}".format(file_name))
                save_batch_image_with_mesh_joints(mesh_renderer, images.to(cpu_device), cam_params.to(cpu_device),
                                                  bboxes.to(cpu_device), est_mesh_cam_xyz, est_pose_uv,
                                                  est_pose_cam_xyz, file_name)

    # overall evaluate pose estimation
    assert len(results_pose_cam_xyz) == len(dataset_val), \
        "The number of estimation results (%d) is inconsistent with that of the ground truth (%d)." % \
        (len(results_pose_cam_xyz), len(dataset_val))

    avg_est_error = dataset_val.evaluate_pose(results_pose_cam_xyz, cfg.EVAL.SAVE_POSE_ESTIMATION, output_dir)  # cm
    logger.info("Overall:\tAverage pose estimation error: {0:.2f} (mm)".format(avg_est_error * 10.0))


if __name__ == "__main__":
    main()
