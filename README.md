# Introduction
HYPO is a physical optics package for modeling and analyzing electromagnetic propagation in optical systems. It is developed to support the study of beam formation and polarization properties in millimeter-wave and sub-millimeter instruments, where diffraction, phase evolution, dielectric-interface effects, and polarization-dependent responses must be treated consistently.

- The framework is intended for three classes of optical systems:

- Purely refractive optics, including dielectric elements and interfaces with anti-reflection (AR) coatings;

- Purely reflective systems, such as reflector antennas and telescope mirrors, planned for future releases;

- Hybrid refractive-reflective systems, in which dielectric and reflective elements jointly determine the final beam, also planned for future releases.

The current version of HYPO focuses on refractive optical systems. A key feature of the package is that it accounts for the electromagnetic response of dielectric interfaces, including the effects of AR coatings. This makes it possible to analyze not only beam intensity and phase distributions, but also polarization-related effects such as polarization-dependent transmission, reflection, and beam distortions introduced by refractive components.

The main purpose of HYPO is to provide a unified framework for studying:
- beam propagation through optical systems,
- near-field and far-field beam patterns,
- amplitude and phase evolution,
- polarization properties and instrumental polarization effects,
- the impact of coated optical surfaces on system performance.

By developing refractive, reflective, and hybrid capabilities within a common physical optics framework, HYPO aims to provide a consistent tool for end-to-end electromagnetic analysis across a wide range of optical designs. In the current release, this capability is implemented for refractive optics, while support for reflective and mixed systems is planned for later versions.

The documentation introduces the scope of the current release, installation steps, core concepts, and example workflows.

# Installation
## Requirements

- Python 3.9 or newer
- NVIDIA GPU
- A compatible NVIDIA driver
- PyTorch with CUDA support

## Python dependencies
- "numpy (>=1.26.4)",
- "scipy (>=1.7)",
- "matplotlib (>=3.5)",
- "torch (>=2.0)",
- "tqdm (>=4.67.3,<5.0.0)"

## Install

Clone the repository and install the package:

```bash
# 1. create environment
conda create -n hypo python=3.10
conda activate hypo

# 2. install pytorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. install this package
git clone https://github.com/XiaodongRencologne/hypo.git
cd hypo
pip install .
```

## Examples

Currently, there are two examples to show how to build a optical system and run po analysis.

### 1. A simple HDPE lens
- [A simple HDPE lens](docs/source/examples/HDPE_SimpleLens.md)
or
- [Jupyter notebook](examples/HDPE_SimpleLens.ipynb)
### 2. Two biconic Silicon lenses
- [Two biconic Silicon lenses](docs/source/examples/Biconic_lens.md)
or

- [Jupyter notebook](examples/Biconic_lens.ipynb)