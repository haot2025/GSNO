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

enable_amp = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dt = 1 * 3600
dt_solver = 150
nsteps = dt // dt_solver

dataset = PdeDataset(
    dt=dt,
    nsteps=nsteps,
    dims=(256, 512),
    num_examples=256,
    device=device,
    normalize=True,
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
)

solver = dataset.solver.to(device)
nlat = dataset.nlat
nlon = dataset.nlon

model = GSNO(
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


def l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    loss = solver.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)

    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)

    return loss.mean()


def train_model(
    model,
    dataloader,
    optimizer,
    nepochs=100,
    nfuture=0,
    num_examples=256,
    num_valid=8,
    load_checkpoint=False,
):
    train_start = time.time()

    exp_dir = "./1h/checkpoints_GSNO"
    os.makedirs(exp_dir, exist_ok=True)

    if load_checkpoint:
        model.load_state_dict(torch.load(os.path.join(exp_dir, "ckpt49.pt")))

    for epoch in range(nepochs):
        epoch_start = time.time()

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(num_examples)

        print(f"Epoch {epoch}: Training with {len(dataloader.dataset)} samples")

        acc_loss = 0.0
        model.train()

        for inp, tar in dataloader:
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=enable_amp):
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)

                loss = l2loss_sphere(solver, prd, tar)

            acc_loss += loss.item() * inp.size(0)

            loss.backward()
            optimizer.step()

        acc_loss /= len(dataloader.dataset)

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(num_valid)

        valid_loss = 0.0
        model.eval()

        with torch.no_grad():
            for inp, tar in dataloader:
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)

                loss = l2loss_sphere(solver, prd, tar, relative=True)
                valid_loss += loss.item() * inp.size(0)

        valid_loss /= len(dataloader.dataset)

        epoch_time = time.time() - epoch_start

        print("--------------------------------------------------------------------------------")
        print(f"Epoch {epoch} summary:")
        print(f"time taken: {epoch_time}")
        print(f"accumulated training loss: {acc_loss}")
        print(f"relative validation loss: {valid_loss}")

        torch.save(model.state_dict(), os.path.join(exp_dir, f"ckpt{epoch}.pt"))

    train_time = time.time() - train_start

    print("--------------------------------------------------------------------------------")
    print(f"done. Training took {train_time}.")

    return valid_loss


torch.manual_seed(666)
if torch.cuda.is_available():
    torch.cuda.manual_seed(666)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0.0)

train_model(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    nepochs=100,
    load_checkpoint=False,
)