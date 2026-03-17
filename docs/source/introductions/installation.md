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