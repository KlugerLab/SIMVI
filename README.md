[![PyPI package](https://img.shields.io/pypi/v/simvi?label=pypi%20package)](https://pypi.org/project/simvi/)
[![Read the Docs](https://img.shields.io/readthedocs/idsc-suite/latest.svg?label=Read%20the%20Docs)](https://idsc-suite.readthedocs.io/)
[![Downloads](https://static.pepy.tech/badge/simvi)](https://pepy.tech/project/simvi)

# SIMVI

SIMVI (Spatial Interaction Modeling using Variational Inference) is a deep generative framework for disentangling intrinsic variation and spatially-induced variation in spatial omics data. SIMVI has rigorous identifiability guarantee, and enables novel downstream analyses, including clustering and differential expression analysis based on disentangled representations, spatial effect identification, spatial effect attribution, and transfer learning on new measurements / modalities. SIMVI is implemented with scvi-tools.

![fig1_github](fig1_github.svg)

Read the SIMVI paper on Nature Communications:
https://www.nature.com/articles/s41467-025-58089-7

Citation:
```
@article{dong2025simvi,
  title={SIMVI disentangles intrinsic and spatial-induced cellular states in spatial omics data},
  author={Dong, Mingze and Su, David G and Kluger, Harriet and Fan, Rong and Kluger, Yuval},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={2990},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
## System requirements
### OS requirements
The SIMVI package is supported for all OS in principle. The package has been tested on the following systems:
* macOS: Monterey (12.4)
* Linux: Ubuntu (20.04.5)
### Dependencies
See `setup.cfg` for details.

## Installation
SIMVI requires `python` version 3.7+.  Install directly from pip with:

    pip install simvi

The installation should take no more than a few minutes on a normal desktop computer.

## Documentation

Please refer to our documentation [here](https://idsc-suite.readthedocs.io/en/latest).

## Usage

For detailed usage, follow our step-by-step tutorial here. The tutorial should take no more than 30 minutes to complete on a standard desktop computer.

- [Getting Started with SIMVI](https://github.com/KlugerLab/SIMVI/blob/main/SIMVI_tutorial_MERFISH.ipynb)

Download the dataset used for the tutorial here:

- [Human MERFISH MTG data](https://drive.google.com/drive/folders/1jeAZge-0wJ1gkHEKC4P6PIalumn2A68p?usp=sharing)
