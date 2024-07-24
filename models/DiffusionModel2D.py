# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

from argparse import ArgumentParser, Namespace

from utils.diffusion import ForwardDiffusion, BackwardDiffusion

# load the backbone network architecture
from models.module_UNet2D_pixelshuffle_inject import (
    module_UNet2D_pixelshuffle_inject as backbone,
)


class DiffusionModel2D(pl.LightningModule):

    def __init__(self, hparams):
        super(DiffusionModel2D, self).__init__()

        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)

        self.network = backbone(
            patch_size=self.hparams.patch_size,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            feat_channels=self.hparams.feat_channels,
            t_channels=self.hparams.t_channels,
            out_activation=self.hparams.out_activation,
            layer_norm=self.hparams.layer_norm,
        )
        # cache for generated images
        self.last_predictions = None
        self.last_imgs = None

        # set up diffusion parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.forward_diffusion = ForwardDiffusion(
            self.hparams.num_timesteps, schedule=self.hparams.diffusion_schedule
        ).to(device)
        self.backward_diffusion = BackwardDiffusion(
            self.network,
            self.hparams.num_timesteps,
            schedule=self.hparams.diffusion_schedule,
            mean_type="epsilon",
            var_type="fixedlarge",
        ).to(device)

    def forward(self, z, t):
        return self.network(z, t)

    def load_pretrained(self, pretrained_file, strict=True, verbose=True):

        # Load the state dict
        state_dict = torch.load(pretrained_file)["state_dict"]

        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)

        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())

        layers = []
        for layer in param_dict:
            if strict and not "network." + layer in state_dict:
                if verbose:
                    print('Could not find weights for layer "{0}"'.format(layer))
                continue
            try:
                param_dict[layer].data.copy_(state_dict["network." + layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print("Error at layer {0}:\n{1}".format(layer, e))

        self.network.load_state_dict(param_dict)

        if verbose:
            print("Loaded weights for the following layers:\n{0}".format(layers))

    def denoise_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):

        # Get image for the current batch
        self.last_imgs = batch["image"].float()

        # Get x_t and noise for a random t
        self.x_t, noise, t = self.forward_diffusion(self.last_imgs)
        self.x_t.requires_grad = True

        # Generate prediction
        self.generated_noise = self.forward(self.x_t, t)

        # Get the losses
        loss_denoise = self.denoise_loss(self.generated_noise, noise)

        self.logger.experiment.add_scalar(
            "loss_denoise", loss_denoise, self.current_epoch
        )

        return loss_denoise

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        x_hat = self.forward(
            x,
            torch.tensor(
                [
                    0,
                ],
                device=x.device.index,
            ),
        )
        return {"test_loss": F.l1_loss(x - x_hat, x)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        x_hat = self.forward(
            x,
            torch.tensor(
                [
                    0,
                ],
                device=x.device.index,
            ),
        )
        return {"val_loss": F.l1_loss(x - x_hat, x)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        opt = torch.optim.RAdam(
            self.network.parameters(), lr=self.hparams.learning_rate
        )
        return [opt], []

    def on_train_epoch_end(self):

        with torch.no_grad():

            input_patch = self.last_imgs

            # Get x_0 which is the prediction from the diffusion model
            x_0 = self.backward_diffusion(input_patch)

            # Ĺog images
            prediction_grid = torchvision.utils.make_grid(x_0)
            self.logger.experiment.add_image(
                "predicted_x_0", prediction_grid, self.current_epoch
            )

            img_grid = torchvision.utils.make_grid(input_patch)
            self.logger.experiment.add_image("raw_x_0", img_grid, self.current_epoch)

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser])

        # network parameters
        parser.add_argument("--in_channels", default=1, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--feat_channels", default=16, type=int)
        parser.add_argument("--t_channels", default=128, type=int)
        parser.add_argument("--patch_size", default=(1, 96, 96), type=int, nargs="+")
        parser.add_argument("--layer_norm", default="instance", type=str)
        parser.add_argument("--out_activation", default="none", type=str)

        # diffusion parameters
        parser.add_argument("--num_timesteps", default=1000, type=int)
        parser.add_argument("--diffusion_schedule", default="cosine", type=str)

        # training parameters
        parser.add_argument("--learning_rate", default=0.001, type=float)

        return parser
