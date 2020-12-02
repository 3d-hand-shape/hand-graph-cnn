from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

from data.utils import get_train_val_im_paths


def main():
    parser = argparse.ArgumentParser(description="List train & validation set of 3D Hand Shape and Pose Dataset")
    parser.add_argument(
        "--image-dir",
        default="./data/synthetic_train_val/images",
        help="path to image directories",
    )
    parser.add_argument(
        "--val-set-path",
        default="./data/synthetic_train_val/3D_labels/val-camera.txt",
        help="path to validation cameras",
    )
    parser.add_argument(
        "--train-val-flag",
        default="val",
        help="train or val",
    )

    args = parser.parse_args()

    image_paths = get_train_val_im_paths(args.image_dir, args.val_set_path,
                                         args.train_val_flag)

    train_val_name = "training" if args.train_val_flag == "train" else "validation"
    print(f"{train_val_name} image paths:")
    for image_path in image_paths:
        print(image_path)

    print(f"\nThere are {len(image_paths)} {train_val_name} images.")


if __name__ == "__main__":
    main()

