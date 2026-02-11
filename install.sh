#!/bin/bash
set -e

# =============================================================================
# PSFA-2022 Installation Script for WSL2/Ubuntu
# References: Dockerfile and README.md
# =============================================================================

echo "Starting installation..."

# 1. System Dependencies
# -----------------------------------------------------------------------------
echo "[1/5] Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        tzdata curl git build-essential cmake pkg-config \
        software-properties-common ffmpeg libboost-all-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libsndfile1 libeigen3-dev libgmp-dev libmpfr-dev
else
    echo "Warning: apt-get not found. Skipping system package installation."
fi

# 2. Conda Environment Setup
# -----------------------------------------------------------------------------
ENV_NAME="psfa"
PYTHON_VERSION="3.9"

echo "[2/5] Configuring Conda environment '$ENV_NAME'..."
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Conda first."
    exit 1
fi

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists. Skipping creation."
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate environment
# Note: This works if the script is sourced, or for the duration of this script
# if we use the hook.
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 3. Python Basic Setup
# -----------------------------------------------------------------------------
echo "[3/5] Installing base Python packages..."
pip install -U "pip<24.1" setuptools==65.7.0 wheel
pip install numpy==1.23.5

# 4. Core Frameworks (PyTorch, Nvdiffrast, TensorFlow)
# -----------------------------------------------------------------------------
echo "[4/5] Installing core AI frameworks..."

# PyTorch 2.0.1 + CUDA 11.8
# Dockerfile uses a local wheel, we fallback to online wheel if missing.
echo "--> Installing PyTorch..."
if [ -f "assets/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl" ]; then
    echo "Using local PyTorch wheel..."
    pip install assets/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl
else
    echo "Downloading PyTorch from index..."
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
fi
pip install torchvision==0.15.2+cu118   torchaudio==2.0.2+cu118   --index-url https://download.pytorch.org/whl/cu118   --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Nvdiffrast
# Requires CUDA toolkit (usually available in WSL2 if drivers are installed on Windows)
echo "--> Installing nvdiffrast..."
export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6"
if [ -f "assets/nvdiffrast.tar.gz" ]; then
    echo "Using local nvdiffrast..."
    pip install --no-build-isolation assets/nvdiffrast.tar.gz
else
    echo "Cloning nvdiffrast from GitHub..."
    pip install git+https://github.com/NVlabs/nvdiffrast.git
fi

# Build dependencies
pip install Cython ninja pybind11

# 5. Project Dependencies
# -----------------------------------------------------------------------------
echo "[5/5] Installing project requirements..."

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi

# TensorFlow & JAX (Matches Dockerfile versions)
echo "--> Installing TensorFlow & JAX..."
pip install tensorflow==2.13.0 jax==0.4.23 jaxlib==0.4.23

# PyTorch Geometric
echo "--> Installing PyTorch Geometric..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch_geometric

# VideoIO
echo "--> Installing videoio..."
pip install videoio==0.3.0

# 6. Model & Assets Check
# -----------------------------------------------------------------------------
echo "[6/6] Checking for required assets..."

ASSETS_DIR="assets"
MISSING_ASSETS=0

# DeepSpeech
if [ ! -d "$ASSETS_DIR/pretrain_models/deepspeech-0.1.0-models" ]; then
    echo "WARNING: DeepSpeech models not found in $ASSETS_DIR/pretrain_models/deepspeech-0.1.0-models"
    echo "  -> Please download from https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz"
    MISSING_ASSETS=1
fi

# FLAME
if [ ! -f "$ASSETS_DIR/flame-data/FLAME2020/generic_model.pkl" ]; then
    echo "WARNING: FLAME model not found at $ASSETS_DIR/flame-data/FLAME2020/generic_model.pkl"
    echo "  -> Please download from https://flame.is.tue.mpg.de/"
    MISSING_ASSETS=1
fi

if [ ! -f "$ASSETS_DIR/flame-data/FLAME_masks/FLAME_masks.pkl" ]; then
    echo "WARNING: FLAME masks not found at $ASSETS_DIR/flame-data/FLAME_masks/FLAME_masks.pkl"
    echo "  -> Please download from https://flame.is.tue.mpg.de/"
    MISSING_ASSETS=1
fi

# FLAME Numpy Conversion Check
if [ -f "$ASSETS_DIR/flame-data/FLAME2020/generic_model.pkl" ] && [ ! -f "$ASSETS_DIR/flame-data/FLAME2020/generic_model-np.pkl" ]; then
    echo "Running FLAME model conversion to numpy..."
    python assets/flame-data/FLAME2020/to_numpy.py
fi


echo "========================================================"
echo "Installation complete!"
if [ $MISSING_ASSETS -eq 1 ]; then
    echo "PLEASE NOTE: Some assets are missing. See warnings above."
fi
echo "Please activate the environment with: conda activate $ENV_NAME"
echo "========================================================"
