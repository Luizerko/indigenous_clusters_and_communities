# Setting Up Environment

Setting up the envinronment is quite straight forward, but you'll have to go through some checks before the installations themselves since we make use of graphics cards for most parts of the project. After that, the creation of the environment and installation of the packasges should run smoothly.

## Graphics Card

First, ensure your graphics card is accessible. Open the terminal and run the command `nvidia-smi`. If this command shows the status of your graphics card and CUDA version, you're ready to proceed. If not, refer to [online tutorials](https://www.cherryservers.com/blog/install-cuda-ubuntu) or the [official documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) to properly install CUDA and `nvcc`.

## Anaconda Environment

If you don't have Anaconda, just install Miniconda (on Linux) through:

```
# Downloading and installing Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Now that you have Miniconda, you just need to close and reopen your terminal to finish up the installation. After that, create your environment with:

```
# Creating Anaconda environment
conda create -n "ind_thesis" python=3.10
```

Once you have your environment, you can start installing the appropriate libraries. In general, this can be done through the `requirements.txt` file, but you'll need to manually install two packages, `torch` and `torchvision`, due to appropriate compatibility with your CUDA version.

Below you'll find the command I ran to install it locally, but you might have to change both the package version or the CUDA version to make it work on your own machine. If this is te case, look for a wheel compatible with your specs at the [torch wheel website](https://download.pytorch.org/whl/torch/) and at the [torchvision wheel website](https://download.pytorch.org/whl/torchvision/).

```
# Installing torch and torchvision
pip install torch==2.5.1+cu121 -f https://download.pytorch.org/whl/torch/
pip install torchvision==0.20.1+cu121 -f https://download.pytorch.org/whl/torchvision/
```

Once these two packages are properly installed, you can simply install all the other necessary packages through:

```
# Installing other dependencies
pip install -r requirements.txt
```

## Expanding Environment

The dependencies of this repo are mainly organized via `pip-tools`. So if you happen to expand it by installing other dependencies, make sure to add the name of the new packages to the `requirements.in` file and then let the tool generate the appropriate `requirements.txt` file through:

```
# Generating new requirements.txt file through pip-tools
pip-compile requirements.in
```