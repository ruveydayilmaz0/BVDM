#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rüveyda Yilmaz
Code adapted from: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011890
"""

import numpy as np
import os
import torch
from skimage import io
from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser
from torch.autograd import Variable
from datasets.HeLa_CTC import HeLaDataset
from utils.diffusion import ForwardDiffusion, BackwardDiffusion
import cv2
from models.DiffusionModel2D import DiffusionModel2D as network


def main(hparams):
    """
    Main testing routine specific for this project
    """

    # Initialize the classes
    model = network(hparams=hparams)
    model = model.load_from_checkpoint(hparams.ckpt_path)
    model = model.cuda()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    forward_diffusion = ForwardDiffusion(
        hparams.num_timesteps, schedule=hparams.diffusion_schedule
    ).to(device)
    backward_diffusion = BackwardDiffusion(
        model,
        hparams.num_timesteps,
        schedule=hparams.diffusion_schedule,
        mean_type="epsilon",
        var_type="fixedlarge",
    ).to(device)
    # Initialize the dataset
    dataset = HeLaDataset(**vars(hparams))

    with open(os.path.join(hparams.masks_path, hparams.masks_info)) as f:
        cells = f.readlines()
    for cell in cells:
        cell = cell.split(" ")
        predicted_img = None
        mask = None
        for mask_id in range(int(cell[1]), int(cell[2][:-1]) + 1):
            print("Processing file: " + str(mask_id))

            # Set the number of diffusion timesteps based on the current frame
            timesteps_start = (
                hparams.timesteps_f0
                if mask_id == int(cell[1])
                else hparams.timesteps_f_non0
            )

            # Create the output folder if non-existent
            os.makedirs(hparams.output_path, exist_ok=True)

            with torch.no_grad():
                # Get the input
                sample = dataset.__getitem__(mask_id, predicted_img, mask)
                data = Variable(
                    torch.from_numpy(sample["mask"][np.newaxis, ...]).cuda()
                )
                data = data.float()
                prediction = data[:, 0:1, ...].clone()
                if timesteps_start != 0:
                    prediction, _, _ = forward_diffusion(
                        prediction, t=timesteps_start - 1
                    )
                    prediction = backward_diffusion(prediction, timesteps_start)

                # Convert final image to numpy
                prediction = prediction.cpu().data.numpy()
                # Remove batch dimension
                prediction = prediction[:, 0, ...]
                mask, predicted_img = sample["plain_mask"], prediction
                # Make cell edges smoother
                mask_tmp = mask[0, ..., None].astype(np.float32)
                mask_tmp[mask_tmp > 0] = 1
                mask_tmp_3d = np.stack((mask_tmp, mask_tmp, mask_tmp), axis=2)[..., 0]
                contours, _ = cv2.findContours(
                    image=mask_tmp.astype(np.uint8),
                    mode=cv2.RETR_TREE,
                    method=cv2.CHAIN_APPROX_NONE,
                )
                cv2.drawContours(
                    image=mask_tmp_3d,
                    contours=contours,
                    contourIdx=-1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                contour = cv2.cvtColor(mask_tmp_3d, cv2.COLOR_BGR2GRAY)
                io.imsave(
                    os.path.join(hparams.output_path, str(mask_id).zfill(4) + ".tif"),
                    np.where(
                        contour <= 1,
                        prediction * np.max(sample["orig_mask"]) / 150,
                        gaussian_filter(prediction, sigma=1)
                        * np.max(sample["orig_mask"])
                        / 150,
                    ),
                )


def get_parser():
    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        "--output_path",
        type=str,
        help="output path for test results",
    )

    parent_parser.add_argument(
        "--masks_path",
        type=str,
        help="path for the synthetic masks",
    )

    parent_parser.add_argument(
        "--masks_info",
        type=str,
        default="label_info.txt",
        help="name of the file that contains information about synthetic masks",
    )

    parent_parser.add_argument(
        "--ckpt_path",
        type=str,
        help="ckpt path",
    )

    parent_parser.add_argument(
        "--gpus", type=int, default=1, help="number of GPUs to use"
    )

    parent_parser.add_argument(
        "--timesteps_f0",
        type=int,
        default=200,
        help="the number of diffusion timesteps for the first frame of each cell",
    )

    parent_parser.add_argument(
        "--timesteps_f_non0",
        type=int,
        default=10,
        help="the number of diffusion timesteps for later frames of each cell",
    )

    return parent_parser


if __name__ == "__main__":

    parent_parser = get_parser()
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()
    main(hyperparams)
