# syntax=docker/dockerfile:1
# Use CUDA 12.1 base image (matches SAM 3D Objects requirements)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge based, no ToS issues)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# Install mamba for faster env creation
RUN conda install -y mamba -c conda-forge

# Initialize conda
RUN conda init bash && \
    echo "conda activate sam3d-objects" >> ~/.bashrc

WORKDIR /app

# Clone SAM 3D Objects first to get their environment file
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /app/sam-3d-objects

# Create conda environment from SAM 3D's exact specification (using mamba for speed)
WORKDIR /app/sam-3d-objects
RUN mamba env create -f environments/default.yml

# Use conda run for all subsequent commands
SHELL ["conda", "run", "-n", "sam3d-objects", "/bin/bash", "-c"]

# Set CUDA architectures for building CUDA extensions (no GPU during docker build)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Set pip extra index URLs
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# Install SAM 3D Objects and dependencies
# Set TORCH_CUDA_ARCH_LIST because there's no GPU during docker build
RUN TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" pip install -e '.[dev]'
RUN TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" pip install -e '.[p3d]'
RUN TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" pip install -e '.[inference]'

# Apply hydra patch
RUN if [ -f ./patching/hydra ]; then chmod +x ./patching/hydra && ./patching/hydra; fi

# Install moge
RUN pip install git+https://github.com/microsoft/MoGe.git

# Install gradio for our app
RUN pip install gradio==4.36.0

# Set CONDA_PREFIX for SAM 3D
ENV CONDA_PREFIX=/opt/conda/envs/sam3d-objects

# Skip import verification - SAM 3D requires GPU at import time
# Verification will happen at runtime when GPU is available
RUN echo "Skipping SAM 3D import verification (no GPU during build)"

WORKDIR /app

# Install our app dependencies
RUN pip install \
    replicate \
    Pillow \
    "trimesh[easy]" \
    rembg \
    onnxruntime \
    requests \
    python-dotenv \
    huggingface_hub

# Copy .env to read HF_TOKEN for checkpoint download, then remove it
COPY .env /tmp/.env
RUN tr -d '\r' < /tmp/.env > /tmp/.env.unix && mv /tmp/.env.unix /tmp/.env && \
    export $(grep -v '^#' /tmp/.env | grep -v '^$' | xargs) && \
    python -c "from huggingface_hub import snapshot_download; import os; snapshot_download('facebook/sam-3d-objects', local_dir='/app/checkpoints', token=os.environ['HF_TOKEN'])" && \
    rm /tmp/.env

# Copy application code
COPY app.py .

# Environment for runtime
ENV SAM3D_REPO_PATH=/app/sam-3d-objects
ENV SAM3D_CHECKPOINT_PATH=/app/checkpoints/hf/pipeline.yaml
ENV PORT=8080

EXPOSE 8080

# Run with conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "sam3d-objects", "python", "app.py"]