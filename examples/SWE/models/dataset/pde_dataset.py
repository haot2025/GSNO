# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from math import ceil

from .shallow_water_equations import ShallowWaterSolver


class PdeDataset(torch.utils.data.Dataset):
    """Custom Dataset class for PDE training data"""

    def __init__(
        self,
        dt,
        nsteps,
        dims=(384, 768),
        grid="equiangular",
        pde="shallow water equations",
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        stream=None,
    ):
        self.num_examples = num_examples
        self.device = device
        self.stream = stream

        self.nlat = dims[0]
        self.nlon = dims[1]

        # number of solver steps used to compute the target
        self.nsteps = nsteps
        self.normalize = normalize

        if pde == "shallow water equations":
            lmax = ceil(self.nlat / 3)
            mmax = lmax
            dt_solver = dt / float(self.nsteps)
            self.solver = ShallowWaterSolver(self.nlat, self.nlon, dt_solver, lmax=lmax, mmax=mmax, grid=grid).to(self.device).float()
        else:
            raise NotImplementedError

        self.set_initial_condition(ictype=initial_condition)

        if self.normalize:
            inp0, _ = self._get_sample()
            self.inp_mean = torch.mean(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            self.inp_var = torch.var(inp0, dim=(-1, -2)).reshape(-1, 1, 1)

    def __len__(self):
        length = self.num_examples if self.ictype == "random" else 1
        return length

    def set_initial_condition(self, ictype="random"):
        self.ictype = ictype

    def set_num_examples(self, num_examples=32):
        self.num_examples = num_examples

    def _get_sample(self):
        if self.ictype == "random":
            inp = self.solver.random_initial_condition(mach=0.2)
        elif self.ictype == "galewsky":
            inp = self.solver.galewsky_initial_condition()

        # solve pde for n steps to return the target
        tar = self.solver.timestep(inp, self.nsteps)
        inp = self.solver.spec2grid(inp)
        tar = self.solver.spec2grid(tar)

        return inp, tar

    def __getitem__(self, index):

        # if self.stream is None:
        #     self.stream = torch.cuda.Stream()

        # with torch.cuda.stream(self.stream):
        #     with torch.inference_mode():
        #         with torch.no_grad():
        #             inp, tar = self._get_sample()

        #             if self.normalize:
        #                 inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
        #                 tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        # self.stream.synchronize()

        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
                    tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        return inp.clone(), tar.clone()
