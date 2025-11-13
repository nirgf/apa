#!/bin/bash

set -e

echo "🚨 Removing any existing CUDA installations..."
sudo apt-get remove --purge '^cuda.*' 'libcudnn*' -y
sudo apt autoremove -y

echo "📦 Installing dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential dkms curl wget gnupg

echo "🔑 Adding CUDA 12.2 repository key..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

echo "⬇️  Installing CUDA 12.2..."
sudo apt-get -y install cuda-toolkit-12-2

echo "✅ CUDA 12.2 installed. Updating PATH..."
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "🧹 Cleaning up install files..."
rm -f cuda-keyring_1.1-1_all.deb

echo "📁 Creating symlink for default CUDA..."
sudo ln -sf /usr/local/cuda-12.2 /usr/local/cuda

echo "⬇️  Downloading cuDNN 9.3.0 archive..."
cd ~
wget https://developer.download.nvidia.com/compute/redist/cudnn/v9.3.0/cudnn-linux-x86_64-9.3.0.29_cuda12-archive.tar.xz

echo "📦 Extracting cuDNN..."
tar -xf cudnn-linux-x86_64-9.3.0.29_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-9.3.0.29_cuda12-archive

echo "🔧 Installing cuDNN 9.3.0 headers and libraries..."
sudo cp include/* /usr/local/cuda/include/
sudo cp lib/* /usr/local/cuda/lib64/
sudo ldconfig

echo "🧹 Cleaning cuDNN install files..."
cd ~
rm -rf cudnn-linux-x86_64-9.3.0.29_cuda12-archive*

echo "✅ Verifying install:"
nvcc --version
grep CUDNN /usr/local/cuda/include/cudnn_version.h | head -3

echo "🎉 CUDA 12.2 + cuDNN 9.3.0 installed successfully!"
echo "👉 You can now install TensorFlow 2.19:"
echo "    pip install tensorflow==2.19"
