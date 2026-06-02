# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 The GSNO Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from models import RealSHT, InverseRealSHT
import os
import matplotlib.pyplot as plt

from ._layers import *
import numpy as np

from functools import partial


class GSNO_Block(nn.Module):

    def __init__(
            self,
            forward_transform,
            inverse_transform,
            input_dim,
            output_dim,
            image_dim,
            mlp_ratio=2.0,
            drop_rate=0.1,
            drop_path=0.1,
            act_layer=nn.GELU,
            norm_layer=nn.Identity,
            inner_skip="linear",
            outer_skip="linear",
            use_mlp=False,
            last_layer=False
    ):
        super().__init__()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        self.global_conv = SpectralConv_GSNO(forward_transform, inverse_transform, input_dim, input_dim, gain=gain_factor, bias=True)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif inner_skip == "identity":
            assert input_dim == output_dim
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # first normalisation layer
        self.norm0 = norm_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        gain_factor = 1.0
        if outer_skip == "linear" or inner_skip == "identity":
            gain_factor /= 2.0

        if use_mlp == True:
            mlp_hidden_dim = int(output_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=input_dim, out_features=output_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                drop_rate=drop_rate, checkpointing=False, gain=gain_factor
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
            torch.nn.init.normal_(self.outer_skip.weight, std=math.sqrt(gain_factor / input_dim))
        elif outer_skip == "identity":
            assert input_dim == output_dim
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        # second normalisation layer
        self.norm1 = norm_layer()

        self.last_layer = last_layer
        self.act = act_layer()

    def forward(self, x):

        x, residual = self.global_conv(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        if self.last_layer == False:
            x = self.act(x)

        return x


class SHNet_GSNO(nn.Module):

    def __init__(
            self,
            img_size=(128, 256),
            grid="equiangular",
            grid_internal="legendre-gauss",
            scale_factor=8,
            in_chans=27,
            out_chans=6,
            embed_dim=64,
            num_layers=4,
            activation_function="gelu",
            encoder_layers=2,
            use_mlp=True,
            mlp_ratio=2.0,
            drop_rate=0.1,
            drop_path_rate=0.1,
            normalization_layer="none",
            hard_thresholding_fraction=1.0,
            use_complex_kernels=True,
            big_skip=True,
            pos_embed=False,
    ):

        super().__init__()

        self.img_size = (64, 128)
        self.img_size_real = img_size
        self.grid = grid
        self.grid_internal = grid_internal
        self.scale_factor = scale_factor
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.encoder_layers = encoder_layers
        self.big_skip = big_skip

        # activation function
        if activation_function == "relu":
            self.activation_function = nn.ReLU
        elif activation_function == "gelu":
            self.activation_function = nn.GELU
        # for debugging purposes
        elif activation_function == "identity":
            self.activation_function = nn.Identity
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # compute downsampled image size. We assume that the latitude-grid includes both poles
        self.h = self.img_size[0] // scale_factor
        self.w = self.img_size[1] // scale_factor

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(nn.LayerNorm, normalized_shape=(self.img_size[0], self.img_size[1]), eps=1e-6)
            norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            norm_layer0 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True,
                                  track_running_stats=False)
            norm_layer1 = partial(nn.InstanceNorm2d, num_features=self.embed_dim, eps=1e-6, affine=True,
                                  track_running_stats=False)
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.")

        if pos_embed == "latlon" or pos_embed == True:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], self.img_size[1]))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "lat":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_size[0], 1))
            nn.init.constant_(self.pos_embed, 0.0)
        elif pos_embed == "const":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, 1, 1))
            nn.init.constant_(self.pos_embed, 0.0)
        else:
            self.pos_embed = None

        # construct an encoder with num_encoder_layers
        num_encoder_layers = self.encoder_layers
        encoder_hidden_dim = int(self.embed_dim * mlp_ratio * 2)
        current_dim = self.in_chans
        encoder_layers = []
        for l in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = encoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.embed_dim, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)
        self.encoder = nn.Sequential(*encoder_layers)

        # compute the modes for the sht
        modes_lat = self.h
        # due to some spectral artifacts with cufft, we substract one mode here
        modes_lon = (self.w // 2 + 1) - 1

        modes_lat = modes_lon = int(min(modes_lat, modes_lon) * self.hard_thresholding_fraction)

        
        ### U-shape architecture for weather forecasting
        self.trans_down1 = RealSHT(*self.img_size_real, lmax=modes_lat * 2, mmax=modes_lon * 2,
                                   grid=self.grid).float()
        self.itrans_down1 = InverseRealSHT(self.h, self.w, lmax=modes_lat * 2, mmax=modes_lon * 2,
                                           grid=grid_internal).float()

        self.trans_mid = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=grid_internal).float()
        self.itrans_mid = InverseRealSHT(self.h // 2, self.w // 2, lmax=modes_lat, mmax=modes_lon,
                                         grid=grid_internal).float()

        self.trans_mid2 = RealSHT(self.h // 2, self.w // 2, lmax=modes_lat // 2, mmax=modes_lon // 2,
                                  grid=grid_internal).float()
        self.itrans_mid2 = InverseRealSHT(self.h, self.w, lmax=modes_lat // 2, mmax=modes_lon // 2,
                                          grid=grid_internal).float()

        self.trans_mid3 = RealSHT(self.h, self.w, lmax=modes_lat, mmax=modes_lon,
                                  grid=grid_internal).float()
        self.itrans_mid3 = InverseRealSHT(*self.img_size_real, lmax=modes_lat, mmax=modes_lon,
                                          grid=grid_internal).float()

        self.trans_up1 = RealSHT(*self.img_size_real, lmax=modes_lat*2, mmax=modes_lon*2,
                                 grid=self.grid_internal).float()
        self.itrans_up1 = InverseRealSHT(*self.img_size_real, lmax=modes_lat*2, mmax=modes_lon*2, grid=grid).float()

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            second_layer = i == 1
            third_layer = i == 2
            last_layer = i == self.num_layers - 1

            if first_layer:
                norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
            elif second_layer:
                norm_layer2 = partial(nn.LayerNorm, normalized_shape=(self.h//2, self.w//2), eps=1e-6)
            elif third_layer:
                norm_layer3 = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
            else:
                norm_layer = nn.Identity

            if first_layer:
                img_dim = 2
            else:
                img_dim = 1

        self.mlp1 = MLP(
            in_features=embed_dim, out_features=embed_dim * 2, hidden_features=embed_dim * 4, act_layer=nn.GELU,
            drop_rate=drop_rate, checkpointing=False, gain=1
        )

        self.dblk = GSNO_Block(
            self.trans_down1,
            self.itrans_down1,
            self.embed_dim * 1,
            self.embed_dim * 2,
            image_dim=img_dim,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=dpr[i],
            act_layer=self.activation_function,
            norm_layer=norm_layer1,
            use_mlp=use_mlp,
        )

        self.mlp2 = MLP(
            in_features=embed_dim * 2, out_features=embed_dim * 4, hidden_features=embed_dim * 8, act_layer=nn.GELU,
            drop_rate=drop_rate, checkpointing=False, gain=1
        )

        self.mblk = GSNO_Block(
            self.trans_mid,
            self.itrans_mid,
            self.embed_dim * 2,
            self.embed_dim * 4,
            image_dim=img_dim,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=dpr[i],
            act_layer=self.activation_function,
            norm_layer=norm_layer2,
            use_mlp=use_mlp,
        )

        self.mblk2 = GSNO_Block(
            self.trans_mid2,
            self.itrans_mid2,
            self.embed_dim * 4,
            self.embed_dim * 4,
            image_dim=img_dim,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=dpr[i],
            act_layer=self.activation_function,
            norm_layer=norm_layer3,
            use_mlp=use_mlp,
        )

        self.mblk3 = GSNO_Block(
            self.trans_mid3,
            self.itrans_mid3,
            self.embed_dim * 4,
            self.embed_dim * 2,
            image_dim=img_dim,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=dpr[i],
            act_layer=self.activation_function,
            norm_layer=norm_layer,
            use_mlp=use_mlp,
        )

        self.upblk = GSNO_Block(
            self.trans_up1,
            self.itrans_up1,
            self.embed_dim * 2,
            self.embed_dim * 1,
            image_dim=img_dim,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=dpr[i],
            act_layer=self.activation_function,
            norm_layer=norm_layer,
            use_mlp=use_mlp,
            last_layer=True
        )

        # construct an decoder with num_decoder_layers
        num_decoder_layers = 1
        decoder_hidden_dim = int(self.embed_dim * mlp_ratio)
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_layers = []
        for l in range(num_decoder_layers - 1):
            fc = nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            # initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)
            decoder_layers.append(fc)
            decoder_layers.append(self.activation_function())
            current_dim = decoder_hidden_dim
        fc = nn.Conv2d(current_dim, self.out_chans, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        decoder_layers.append(fc)
        self.decoder = nn.Sequential(*decoder_layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def down_features1(self, x):

        x = self.encoder(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        residual1 = x

        x = self.dblk(x)

        return x, residual1

    def mid_features1(self, x):

        x = self.mblk(x)

        return x

    def mid_features2(self, x):

        x = self.mblk2(x)

        return x

    def mid_features3(self, x):

        x = self.mblk3(x)

        return x

    def up_features1(self, x):

        x = self.upblk(x)

        return x

    def forward(self, x, y=None, lead_times=None, variables=None, out_variables=None, metric=None, lat=None):

        if self.big_skip:
            residual = x

        x, residual1 = self.down_features1(x)
        residual1 = self.mlp1(residual1)
        residual2 = self.mlp2(x)

        x = self.mid_features1(x)

        x = self.mid_features2(x)

        x = self.mid_features3(x + residual2)

        x = self.up_features1(x + residual1)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        preds = self.decoder(x)

        if metric is not None and y is not None:
            loss = [m(preds, y, out_variables, lat) for m in metric]
        else:
            loss = None

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix,
                 save_plots=False):
        
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat)

        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]