# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 The GSNO Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.utils.checkpoint import checkpoint
import math

from models import *


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features = None,
                 out_features = None,
                 act_layer = nn.ReLU,
                 output_bias = False,
                 drop_rate = 0.,
                 checkpointing = False,
                 gain = 1.0):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Fist dense layer
        fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(fc1.weight, mean=0., std=scale)
        if fc1.bias is not None:
            nn.init.constant_(fc1.bias, 0.0)

        # activation
        act = act_layer()

        # output layer
        fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(fc2.weight, mean=0., std=scale)
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.:
            drop = nn.Dropout2d(drop_rate)
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)
        else:
            self.fwd = nn.Sequential(fc1, act, fc2)

    @torch.jit.ignore
    def checkpoint_forward(self, x):
        return checkpoint(self.fwd, x)

    def forward(self, x):
        if self.checkpointing:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)


def batched_spherical_integral(f):

    B1, B2, Nθ, Nφ = f.shape

    theta = torch.linspace(0, torch.pi, Nθ, device=f.device, dtype=f.dtype)
    phi = torch.linspace(0, 2 * torch.pi, Nφ, device=f.device, dtype=f.dtype)

    dθ = theta[1] - theta[0]
    dφ = phi[1] - phi[0]

    sin_theta = torch.sin(theta).view(1, 1, Nθ, 1)
    weights = sin_theta * dθ * dφ

    integral = torch.sum(f * weights, dim=(-2, -1), keepdim=True)
    average = integral / (4 * torch.pi)

    return average

    
class SpectralConv_GSNO(nn.Module):
    """
    Modified Spectral Convolution according to Driscoll & Healy. Designed for convolutions on the two-sphere S2
    using the Spherical Harmonic Transforms in torch-harmonics, but supports convolutions on the periodic
    domain via the RealFFT2 and InverseRealFFT2 wrappers.
    """

    def __init__(self,
                 forward_transform,
                 inverse_transform,
                 in_channels,
                 out_channels,
                 image_dim,
                 gain = 2.,
                 operator_type = "driscoll-healy",
                 lr_scale_exponent = 0,
                 bias = False):
        super().__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (self.forward_transform.nlat != self.inverse_transform.nlat) \
                        or (self.forward_transform.nlon != self.inverse_transform.nlon)

        # remember factorization details
        self.operator_type = operator_type

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [in_channels, in_channels]
        weight_shape_n = [in_channels] + [self.modes_lat, self.modes_lon]

        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat, self.modes_lon]
            self.contract_func = "...ilm,oilm->...olm"
        elif self.operator_type == "block-diagonal":
            weight_shape += [self.modes_lat, self.modes_lon, self.modes_lon]
            self.contract_func = "...ilm,oilnm->...oln"
        elif self.operator_type == "driscoll-healy":
            weight_shape += [self.modes_lat]
            self.contract_func = "...ilm,oil->...olm"
        else:
            raise NotImplementedError(f"Unkonw operator type f{self.operator_type}")



        # form weight tensors
        scale = math.sqrt(gain / in_channels)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))
        self.weight1 = nn.Parameter(scale * torch.randn(*weight_shape_n, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))


    def forward(self, x):

        dtype = x.dtype
        x = x.float()
        residual = x
        b,c,h,w = x.shape
        x_int = batched_spherical_average(x) #simplied spherical integral

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.forward_transform(x)#coefficients
            if self.scale_residual:
                residual = self.inverse_transform(x)

        w1 = self.weight1.unsqueeze(0).expand(b, -1, -1, -1)
        x_c = x_int * w1
        x = x + x_c
        x = torch.einsum(self.contract_func, x, self.weight)

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias
        x = x.type(dtype)

        return x, residual
