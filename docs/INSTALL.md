# Setting Up the Environment

Setting up the environment is straightforward, but before proceeding with installations, you need to ensure your graphics card is properly configured. Since most parts of this project utilize GPU acceleration, **verifying CUDA compatibility** is crucial. Once the prerequisites are met, the environment creation and package installations should proceed smoothly.

## Graphics Card Setup

First, confirm that your graphics card is accessible by running `nvidia-smi` in the terminal. If this command displays your GPU status and CUDA version, you’re all set to proceed. Otherwise, you may need to install CUDA and `nvcc`. You can follow these resources for guidance:

- [CUDA Installation Guide for Ubuntu](https://www.cherryservers.com/blog/install-cuda-ubuntu)

- [NVIDIA Official Documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## Installing Anaconda / Miniconda

If you don’t have *Anaconda* installed, you can install *Miniconda* (a lightweight alternative) on Linux using the following commands:

```
# Downloading and installing Miniconda

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installation, close and reopen your terminal to finalize the setup.

### Creating the Conda Environment

Once *Miniconda* is installed, create a new `conda` environment with `Python 3.10`:

```
# Creating conda environment

conda create -n "ind_thesis" python=3.10
```

## Installing Dependencies

The required dependencies are listed in `requirements.txt`, but before installing them, you must **manually install `torch` and `torchvision`** due to CUDA compatibility constraints.

### Installing `torch` and `torchvision`

The command below installs `torch` and `torchvision` with CUDA 12.1 support:

```
# Installing torch and torchvision with CUDA 12.1 support

pip install torch==2.5.1+cu121 -f https://download.pytorch.org/whl/torch/
pip install torchvision==0.20.1+cu121 -f https://download.pytorch.org/whl/torchvision/
```

Since your CUDA version may differ, if the installation fails, find the correct package versions from:

- [Torch Wheel Repository](https://download.pytorch.org/whl/torch/)

- [Torchvision Wheel Repository](https://download.pytorch.org/whl/torchvision/)

Alternatively, you can build it from source, but this is not recommended due to time and complexity.

### Installing Other Dependencies

Once `torch` and `torchvision` are installed, install the remaining dependencies from `requirements.txt`:

```
# Installing other dependencies

pip install -r requirements.txt
```


## Expanding the Environment

The dependencies in this repository are managed using `pip-tools`. If you need to add new dependencies, first add the new package names to `requirements.in` and then run the following command to update `requirements.txt` accordingly:

```
# Generating new requirements.txt file through pip-tools

pip-compile requirements.in
```