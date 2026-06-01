# Green's-Function Spherical Neural Operators


This repository contains the official implementation of **Generalized Spherical Neural Operators: Green's Function Formulation**  [Paper Link](https://openreview.net/forum?id=XkGjzSDTnm) (ICLR 2026)

## Overview
- Green's-function Spherical Neural Operator (GSNO): a operator block adding absolute position-dependent Green's function.
- Spherical Harmonic Neural Operator Network (SHNet): a hierarchical U-shape architecture

## Repository Structure

```
GSNO/
│
├── models/            
│   ├── SHNet.py             # GSNO and SHNet
│   └── _layers.py              # Basic spectral layers
│
├── examples/
│   ├── SWE/                 # Spherical Shallow Water Equation experiments
│
├── requirements.txt
└── README.md
```

## Datasets and Benchmarks

All datasets and benchmark implementations used in this work are publicly available from their official repositories.

| Benchmark | Source |
|---|---|
| WeatherBench | https://github.com/pangeo-data/WeatherBench |
| Shallow Water Equation | https://github.com/NVIDIA/torch-harmonics |

We thank the authors of these datasets and benchmarks for making their resources publicly available.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{
tang2026generalized,
title={Generalized Spherical Neural Operators: Green{\textquoteright}s Function Formulation},
author={Hao Tang and Hao Chen and Chao Li},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=XkGjzSDTnm}
}
```
