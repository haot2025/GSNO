# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py

from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms
import numpy

from models.GSNO.gsno import SHNet_GSNO
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    l2loss_sphere,
)
from utils.pos_embed import interpolate_pos_embed


class GlobalForecastModule(LightningModule):

    def __init__(
        self,
        net: SHNet_GSNO,
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch #train

        loss_dict, _ = self.net.forward(x, y, lead_times, variables, out_variables, [l2loss_sphere], lat=self.lat)
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict["loss"]

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=log_postfix,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
            save_plots=(batch_idx == 25),
        )

        if not hasattr(self, "test_loss_accumulator"):
            from collections import defaultdict
            self.test_loss_accumulator = defaultdict(list)

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                v = d[k]

                if isinstance(v, torch.Tensor):
                    scalar_value = v.detach().cpu().item()
                elif isinstance(v, numpy.ndarray):
                    scalar_value = v.item()
                else:
                    scalar_value = v 

                self.test_loss_accumulator[k].append(scalar_value)
                loss_dict[k] = scalar_value

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def on_test_epoch_end(self):
        import numpy as np

        if not hasattr(self, "test_loss_accumulator"):
            return

        stats_dict = {}
        for key, values in self.test_loss_accumulator.items():
            values_np = np.array(values)
            mean = np.mean(values_np)
            std = np.std(values_np)
            stats_dict[f"{key}_mean"] = mean
            stats_dict[f"{key}_std"] = std
            stats_dict[f"{key}"] = f"{mean:.3f} ± {std:.3f}"

        for key in stats_dict:
            if key.endswith("_mean") or key.endswith("_std"):
                self.log(
                    f"test_summary/{key}",
                    stats_dict[key],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        print("\nTest Results (Mean ± Std):")
        for key in self.test_loss_accumulator.keys():
            print(f"{key:25}: {stats_dict[f'{key}']}")

        self.test_loss_accumulator.clear()

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
