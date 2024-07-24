# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import random
from utils.warp_texture import Vxm
import glob
import copy


class HeLaDataset(Dataset):
    """
    Dataset of fluorescently labeled cell membranes
    """

    def __init__(
        self,
        data_path="",
        train=False,
        **kwargs,
    ):
        self.train = train

        if not self.train:
            self.masks_path = kwargs["masks_path"]
            self.vxm = Vxm(kwargs)
        else:
            self.data_path = data_path
            print("Getting statistics from images...")
            self.data_statistics = {
                "min": [],
                "max": [],
            }
            # this normalization is crucial for limiting the values in range (-1,1)
            # otherwise, the values might be too high (even around 3500 in the real images)
            for file in self.data_path.dataset[:20]:
                full_path = random.choice(glob.glob(file + "/*"))
                # pick a random image for each cell
                image = io.imread(full_path).astype(np.float32)
                self.data_statistics["min"].append(np.min(image))
                self.data_statistics["max"].append(np.max(image))

            # Construct data set statistics
            self.data_statistics["min"] = np.min(self.data_statistics["min"])
            self.data_statistics["max"] = np.max(self.data_statistics["max"])

            # Get the normalization values
            self.norm1 = self.data_statistics["min"]
            self.norm2 = self.data_statistics["max"] - self.data_statistics["min"]

    def __len__(self):
        return len(self.data_path.dataset)

    def _normalize(self, data):

        data -= self.norm1
        data /= self.norm2
        # minmax data normalization
        data = np.clip(data, 1e-5, 1)
        # data = np.clip(data, 0.08, 1)
        return data

    def __getitem__(self, idx, prev_output=None, prev_mask=None):
        sample = {}

        if self.train:
            # Get the path to the image
            filepath = self.data_path.dataset[idx]

            # There are 2 things to select from the dataset: the cell and the frame
            # cell was already selected by idx, now pick the frame
            img_path = random.choice(glob.glob(filepath + "/*"))
            image = io.imread(img_path)
            image = image[None, ...]
            image = image.astype(np.float32)
            image = self._normalize(image)
            sample["image"] = image

        else:
            # mask: is the resulting mask after warping(plain mask if it is the 1st frame)
            # plain_mask: the original mask itself

            # Read the mask
            plain_mask = io.imread(
                os.path.join(self.masks_path, str(idx).zfill(4)) + ".png"
            )
            orig_mask = copy.deepcopy(plain_mask)
            plain_mask[plain_mask > 0] = 150
            plain_mask = plain_mask / 255
            plain_mask = plain_mask[None, ...]

            # for frames>0 warp previous mask frame according to current and apply it to current
            if prev_output is not None:
                mask = self.vxm.warp(prev_output, prev_mask, plain_mask)[..., 0]
            else:
                mask = plain_mask

            sample["mask"] = mask
            sample["plain_mask"] = plain_mask
            sample["orig_mask"] = orig_mask
        return sample
