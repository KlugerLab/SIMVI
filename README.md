# SIMVI

SIMVI (Spatial Interaction Modelling using Variational Inference) is a deep generative framework for disentangling intrinsic variation and spatially-induced variation in spatial omics data. SIMVI has rigorous identifiability guarantee, and enables novel downstream analyses, including clustering and differential expression analysis based on disentangled representations, spatial effect identification, spatial effect attribution, and transfer learning on new measurements / modalities. SIMVI is implemented with scvi-tools.

![fig1_github](https://github.com/user-attachments/assets/aaa16812-99e7-4f43-ace8-363c2d6dceeb)

Read our updated preprint on BioRxiv:

## System requirements
### OS requirements
The SIMVI package is supported for all OS in principle. The package has been tested on the following systems:
* macOS: Monterey (12.4)
* Linux: Ubantu (20.04.5)
### Dependencies
See `setup.cfg` for details.

## Installation
SIMVI requires `python` version 3.7+.  Install directly from pip with:

    pip install simvi

The installation should take no more than a few minutes on a normal desktop computer.


## Usage

For detailed usage, follow our step-by-step tutorial here:

- [Getting Started with SIMVI](https://github.com/MingzeDong/SIMVI/blob/main/SIMVI_tutorial_MERFISH.ipynb)

Download the dataset used for the tutorial here:

- [Human MERFISH MTG data](https://drive.google.com/drive/folders/1jeAZge-0wJ1gkHEKC4P6PIalumn2A68p?usp=sharing)
