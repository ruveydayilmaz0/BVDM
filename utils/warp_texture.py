#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rüveyda Yilmaz
Code adapted from: https://ieeexplore.ieee.org/abstract/document/8633930
"""

import numpy as np
import voxelmorph as vxm
import tensorflow as tf


class Vxm:

    def __init__(self, kwargs):

        # load the pretrained model
        self.vxm_model = self.vxm_model = self.create_vxm(kwargs["patch_size"][1:])

    # load pretrained model
    def create_vxm(
        self,
        vol_shape,
        checkpoint_path="voxelmorph_ckpt/cp.ckpt",
    ):
        # configure unet features
        nb_features = [
            [32, 32, 32, 32],  # encoder features
            [32, 32, 32, 32, 32, 16],  # decoder features
        ]
        # unet
        vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)
        # losses and loss weights
        losses = ["mse", vxm.losses.Grad("l2").loss]
        loss_weights = [1, 0.01]
        vxm_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=losses,
            loss_weights=loss_weights,
        )
        vxm_model.load_weights(checkpoint_path).expect_partial()
        return vxm_model

    def warp(self, base_img, base_mask, target_mask):
        ratio = np.max(target_mask) / np.max(base_mask)
        base_mask = base_mask / np.max(base_mask)
        target_mask = target_mask / np.max(target_mask)
        base_mask = base_mask[..., None]
        target_mask = target_mask[..., None]
        val_input = [base_mask, target_mask]
        val_pred = self.vxm_model.predict(val_input)
        moved = vxm.networks.Transform(
            base_img[0, ...].shape, interp_method="nearest"
        ).predict([base_img[..., None], val_pred[1]])
        moved[np.invert(target_mask.astype(bool))] = 0
        moved[target_mask.astype(bool)] *= ratio
        return moved
