# SIMVI

SIMVI (Spatial Interaction Modelling using Variational Inference) is a deep generative framework for disentangling intrinsic variation and spatially-induced variation in spatial omics data. SIMVI enables novel downstream analyses, including clustering and differential expression analysis based on disentangled representations, spatial effect identification, spatial effect attribution, and transfer learning on new measurements / modalities. SIMVI is implemented with scvi-tools.

![fig1_rep](https://github.com/MingzeDong/SIMVI/assets/68533876/88a10941-afab-440d-8fd5-95a8ef968d9d)

Read our preprint on BioRxiv: 

- Dong, Mingze, et al. "SIMVI reveals intrinsic and spatial-induced states in spatial omics data". bioRxiv (2023). [Link](https://www.biorxiv.org/content/10.1101/2023.08.28.554970v1)

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

For detailed usage, follow our step-by-step tutorial here. The tutorial should take several minutes to compile, depending on the computational capacity.

- [Getting Started with SIMVI](https://github.com/KlugerLab/SIMVI/blob/main/SIMVItutorial.ipynb)

Download the dataset used for the tutorial here:

- [Human MERFISH MTG data](https://drive.google.com/file/d/1i6spfxfEqqczgSHDX0gNImrGkH7Ruy7z/view?usp=sharing)
