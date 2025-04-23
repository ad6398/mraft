#!/usr/bin/env bash
set -e

# 1. Update and install essential OS packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    dkms \
    linux-headers-$(uname -r) \
    wget \
    gnupg \
    software-properties-common \
    git \
    python3-pip \
    python3-venv \
    ca-certificates

# 2. Install NVIDIA driver (will install latest recommended)
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-driver-525

# 3. Download & install CUDA 11.8 toolkit
CUDA_VERSION="11-8"
CUDA_PKG="cuda-repo-ubuntu2204-${CUDA_VERSION}_local.deb"

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_PKG}
sudo dpkg -i ${CUDA_PKG}
sudo cp /var/cuda-repo-ubuntu2204-${CUDA_VERSION}_local/7fa2af80.pub /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y cuda

# 4. Install cuDNN libraries
sudo apt-get install -y libcudnn8 libcudnn8-dev

# 5. Update your shell environment to include CUDA
if ! grep -q '/usr/local/cuda/bin' ~/.bashrc; then
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
# Reload .bashrc for this session
source ~/.bashrc

# 6. Upgrade pip and install Python libraries
python3 -m pip install --upgrade pip
python3 -m pip install \
    git+https://github.com/illuin-tech/colpali \
    "transformers>4.45.0" \
    "bitsandbytes>=0.45.5" \
    datasets

echo "All done! CUDA, cuDNN, drivers, and your Python packages are installed."
