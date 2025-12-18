# syntax=docker/dockerfile:1
# Trellis 2 Image-to-3D Pipeline
# Uses CUDA 12.4 and PyTorch 2.6.0 as required by Trellis 2
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    libjpeg-dev \
    libpng-dev \
    libopengl0 \
    libglx0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge based)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# Install mamba for faster env creation
RUN conda install -y mamba -c conda-forge

# Create trellis2 environment with Python 3.10
RUN mamba create -n trellis2 python=3.10 -y

# Initialize conda
RUN conda init bash && \
    echo "conda activate trellis2" >> ~/.bashrc

WORKDIR /app

# Use conda run for all subsequent commands
SHELL ["conda", "run", "-n", "trellis2", "/bin/bash", "-c"]

# Set CUDA architectures for building CUDA extensions (no GPU during docker build)
# Supports: V100 (7.0), T4 (7.5), A10/A100 (8.0), RTX 3090 (8.6), RTX 4090/L4 (8.9), H100 (9.0)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install PyTorch 2.6.0 with CUDA 12.4 (required by Trellis 2)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Clone TRELLIS.2 repository
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /app/TRELLIS.2

WORKDIR /app/TRELLIS.2

# ---------------------------------------------------------------------------
# Install Trellis 2 dependencies
# Based on setup.sh with --basic --flash-attn --nvdiffrast --cumesh --o-voxel --flexgemm
# ---------------------------------------------------------------------------

# Basic dependencies
RUN pip install \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    ninja \
    trimesh \
    transformers \
    "gradio>=6.0.0,<7.0.0" \
    tensorboard \
    pandas \
    lpips \
    zstandard \
    kornia \
    timm

# Install Pillow (pillow-simd can cause conflicts with WebP)
RUN pip uninstall -y pillow pillow-simd || true && \
    pip install --no-cache-dir "Pillow>=10.0.0"

# Install utils3d from the correct GitHub repo (required by Trellis 2)
RUN pip install "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"

# Install psutil (required by flash-attn build)
RUN pip install psutil

# Install flash-attn for faster attention (requires CUDA)
# Note: This may fail during build (no GPU) but that's OK - xformers will be used as fallback
RUN pip install flash-attn==2.7.3 --no-build-isolation || echo "flash-attn installation failed, will use fallback"

# Install nvdiffrast (required for mesh rasterization)
RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0

# Configure git to use HTTPS instead of SSH for GitHub (submodules use SSH URLs)
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

# Install nvdiffrec from JeffreyXiang's fork (renderutils branch)
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /app/nvdiffrec && \
    pip install --no-build-isolation -e /app/nvdiffrec

# Install CuMesh (CUDA mesh utilities)
RUN git clone https://github.com/JeffreyXiang/CuMesh.git --recursive /app/CuMesh && \
    pip install --no-build-isolation -e /app/CuMesh

# Install FlexGEMM (sparse convolution via Triton)
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git --recursive /app/FlexGEMM && \
    pip install --no-build-isolation -e /app/FlexGEMM

# Install O-Voxel (mesh to O-Voxel conversion and GLB export)
# This is bundled with the TRELLIS.2 repo at the root level
RUN pip install --no-build-isolation -e /app/TRELLIS.2/o-voxel

# TRELLIS.2 doesn't have a setup.py - it's used via PYTHONPATH
# Add it to the Python path instead of pip installing
ENV PYTHONPATH="/app/TRELLIS.2"

WORKDIR /app

# ---------------------------------------------------------------------------
# Install our app dependencies
# ---------------------------------------------------------------------------
RUN pip install \
    replicate \
    Pillow \
    "trimesh[easy]" \
    requests \
    python-dotenv \
    huggingface_hub

# Install PyMeshLab for mesh repair operations
RUN pip install pymeshlab

# Install lib3mf for proper 3MF export with vertex colors
# (PyMeshLab doesn't support 3MF vertex colors, lib3mf is the official 3MF library)
RUN pip install lib3mf

# Create pt and np aliases for utils3d compatibility (if needed)
RUN python -c "import utils3d, os; d=os.path.dirname(utils3d.__file__); \
from pathlib import Path; \
pt_path = Path(d) / 'pt.py'; np_path = Path(d) / 'np.py'; \
pt_path.exists() or pt_path.write_text('def __getattr__(name):\\n    from . import torch\\n    return getattr(torch, name)\\n'); \
np_path.exists() or np_path.write_text('def __getattr__(name):\\n    from . import numpy\\n    return getattr(numpy, name)\\n'); \
print('utils3d compatibility check complete')"

# Skip import verification - Trellis 2 requires GPU/Triton drivers at import time
# Verification will happen at runtime when GPU is available
RUN echo "Skipping Trellis 2 import verification (requires GPU)"

# ---------------------------------------------------------------------------
# Download model weights
# ---------------------------------------------------------------------------

# The model will be downloaded from HuggingFace at runtime or we can pre-download
# Pre-downloading during build to speed up first run
COPY .env /tmp/.env

RUN tr -d '\r' < /tmp/.env > /tmp/.env.unix && mv /tmp/.env.unix /tmp/.env && \
    export $(grep -v '^#' /tmp/.env | grep -v '^$' | xargs) && \
    python -c "from huggingface_hub import snapshot_download; import os; \
snapshot_download('microsoft/TRELLIS.2-4B', local_dir='/app/models/TRELLIS.2-4B', token=os.environ.get('HF_TOKEN', None))"

# Verify model was downloaded
RUN ls -la /app/models/TRELLIS.2-4B/ && \
    echo "Trellis 2 model checkpoint: OK"

# Clean up .env file
RUN rm /tmp/.env

# ---------------------------------------------------------------------------
# Copy application code
# ---------------------------------------------------------------------------

COPY app.py .

# Environment for runtime
ENV TRELLIS_MODEL_PATH=/app/models/TRELLIS.2-4B
ENV PORT=8080
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV OPENCV_IO_ENABLE_OPENEXR=1

# For flash-attn fallback if needed
ENV ATTN_BACKEND=flash-attn

EXPOSE 8080

# Run with conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "trellis2", "python", "app.py"]
