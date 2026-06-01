# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 The GSNO Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import time
import torch
import numpy as np

from models import RealSHT
from models.dataset import PdeDataset
from models.GSNO.gsno import (
    GSNO_Net as GSNO,
)

torch.manual_seed(333)
if torch.cuda.is_available():
    torch.cuda.manual_seed(333)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)

dt = 1 * 3600
dt_solver = 150
nsteps = dt // dt_solver

grid = "legendre-gauss"
nlat, nlon = 256, 512

dataset = PdeDataset(
    dt=dt,
    nsteps=nsteps,
    dims=(nlat, nlon),
    device=device,
    grid=grid,
    normalize=True,
)

dataset.sht = RealSHT(
    nlat=nlat,
    nlon=nlon,
    grid=grid,
).to(device=device)

nlat = dataset.nlat
nlon = dataset.nlon


def l2loss_sphere_total(solver, prd, tar, relative=False, squared=False):
    loss = solver.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)

    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)

    return loss.mean()


def l2loss_sphere_channels(solver, prd, tar, relative=False, squared=False):
    squared_error = (prd - tar) ** 2
    loss = solver.integrate_grid(squared_error, dimensionless=True)

    if relative:
        denominator = solver.integrate_grid(tar**2, dimensionless=True)

        loss_total = loss.sum(dim=-1) / denominator.sum(dim=-1)
        loss_channel_1 = loss[0, 0] / denominator[0]
        loss_channel_2 = loss[0, 1] / denominator[1]
        loss_channel_3 = loss[0, 2] / denominator[2]
    else:
        loss_total = loss.sum(dim=-1)
        loss_channel_1 = loss[0, 0]
        loss_channel_2 = loss[0, 1]
        loss_channel_3 = loss[0, 2]

    if not squared:
        loss_total = torch.sqrt(loss_total)
        loss_channel_1 = torch.sqrt(loss_channel_1)
        loss_channel_2 = torch.sqrt(loss_channel_2)
        loss_channel_3 = torch.sqrt(loss_channel_3)

    return loss_total, loss_channel_1, loss_channel_2, loss_channel_3


def autoregressive_inference(
    model,
    dataset,
    nsteps,
    autoreg_steps=5,
    nics=50,
):
    model.eval()

    losses = np.zeros(nics)
    losses_total = np.zeros(nics)
    losses_channels_1 = np.zeros(nics)
    losses_channels_2 = np.zeros(nics)
    losses_channels_3 = np.zeros(nics)
    fno_times = np.zeros(nics)

    for iic in range(nics):
        ic = dataset.solver.random_initial_condition(mach=0.2)

        inp_mean = dataset.inp_mean
        inp_var = dataset.inp_var

        prd = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
        prd = prd.unsqueeze(0)

        uspec = ic.clone()

        start_time = time.time()

        for _ in range(autoreg_steps):
            prd = model(prd)
            uspec = dataset.solver.timestep(uspec, nsteps)

        fno_times[iic] = time.time() - start_time

        ref = dataset.solver.spec2grid(uspec)
        prd = prd * torch.sqrt(inp_var) + inp_mean

        losses[iic] = l2loss_sphere_total(
            dataset.solver,
            prd,
            ref,
            relative=True,
        ).item()

        loss_total, loss_channel_1, loss_channel_2, loss_channel_3 = (
            l2loss_sphere_channels(
                dataset.solver,
                prd,
                ref,
                relative=True,
            )
        )

        losses_total[iic] = loss_total.item()
        losses_channels_1[iic] = loss_channel_1.item()
        losses_channels_2[iic] = loss_channel_2.cpu().item()
        losses_channels_3[iic] = loss_channel_3.cpu().item()

    return (
        losses_total,
        losses_channels_1,
        losses_channels_2,
        losses_channels_3,
        losses,
        fno_times,
    )


exp_dir = "./1h/checkpoints_GSNO"
ckpt_path = os.path.join(exp_dir, "ckpt49.pt")

model = FaNO(
    img_size=(nlat, nlon),
    grid="equiangular",
    num_layers=4,
    scale_factor=3,
    embed_dim=256,
    big_skip=True,
    pos_embed=False,
    use_mlp=True,
    normalization_layer="none",
).to(device)

print(
    "Num of params",
    sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad),
)

model.load_state_dict(torch.load(ckpt_path, map_location=device))

metrics = {}

with torch.inference_mode():
    (
        losses_total,
        losses_channels_1,
        losses_channels_2,
        losses_channels_3,
        losses,
        fno_times,
    ) = autoregressive_inference(
        model=model,
        dataset=dataset,
        nsteps=nsteps,
        autoreg_steps=5,
        nics=50,
    )

metrics["loss_mean"] = np.mean(losses)
metrics["loss_std"] = np.std(losses)
metrics["fno_time_mean"] = np.mean(fno_times)
metrics["fno_time_std"] = np.std(fno_times)

print(metrics["loss_mean"])
print(metrics["loss_std"])

print(np.mean(losses_total))
print(np.std(losses_total))

print(np.mean(losses_channels_1))
print(np.std(losses_channels_1))

print(np.mean(losses_channels_2))
print(np.std(losses_channels_2))

print(np.mean(losses_channels_3))
print(np.std(losses_channels_3))

print(f"Channel 1: {np.mean(losses_channels_1):.6f} ± {np.std(losses_channels_1):.6f}")
print(f"Channel 2: {np.mean(losses_channels_2):.6f} ± {np.std(losses_channels_2):.6f}")
print(f"Channel 3: {np.mean(losses_channels_3):.6f} ± {np.std(losses_channels_3):.6f}")

all_losses = np.concatenate(
    [losses_channels_1, losses_channels_2, losses_channels_3]
)

print(f"Overall : {np.mean(all_losses):.6f} ± {np.std(all_losses):.6f}")