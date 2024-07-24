#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rüveyda Yilmaz
Code adapted from: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011890
"""

from argparse import ArgumentParser

import numpy as np
import torch
import glob
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.dataModule import DataModule
from models.DiffusionModel2D import DiffusionModel2D as network

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    # Create the model
    model = network(hparams=hparams)
    os.makedirs(hparams.ckpt_path, exist_ok=True)

    # Resume from checkpoint if available
    resume_ckpt = None
    if hparams.resume:
        checkpoints = glob.glob(os.path.join(hparams.ckpt_path, "*.ckpt"))
        checkpoints.sort(key=os.path.getmtime)
        if len(checkpoints) > 0:
            resume_ckpt = checkpoints[-1]
            print("Resuming from checkpoint: {0}".format(resume_ckpt))

    # Initialize the model
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.ckpt_path,
        filename="Diffusion2D" + "-{epoch:03d}-{step}",
        save_top_k=1,
        monitor="step",
        mode="max",
        verbose=True,
        every_n_epochs=1,
    )

    logger = TensorBoardLogger(
        save_dir=hparams.log_path, name="lightning_logs_diffusion2d"
    )

    trainer = Trainer(
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        resume_from_checkpoint=resume_ckpt,
    )

    data_module = DataModule(hparams.batch_size, hparams.data_root)

    # Train the model
    trainer.fit(model, data_module)


def get_parser():
    parent_parser = ArgumentParser(add_help=False)

    # training parameters
    parent_parser.add_argument(
        "--ckpt_path",
        type=str,
        help="path to save the trained model",
    )

    parent_parser.add_argument(
        "--log_path",
        type=str,
        help="output path for test results",
    )

    parent_parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of GPUs to train on (int) or which GPUs to \
            train on (list or str)",
    )

    parent_parser.add_argument("--epochs", type=int, default=1, help="number of epochs")

    parent_parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Do not resume training from latest checkpoint",
    )

    # data parameters
    parent_parser.add_argument(
        "--data_root",
        type=str,
    )

    parent_parser.add_argument("--batch_size", default=32, type=int)

    return parent_parser


if __name__ == "__main__":

    parent_parser = get_parser()
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()
    main(hyperparams)
