# text-to-3d-printer

Generate 3D printable models from text descriptions using Microsoft Trellis 2.

**Pipeline:**
1. Text prompt → Generate images (Nano Banana via Replicate)
2. Select best image
3. Auto-segment object from background (rembg)
4. Reconstruct 3D model with PBR materials (Trellis 2)
5. Download GLB, STL, or 3MF for 3D printing

**Output Formats:**
- **GLB/GLTF** - Full color with PBR materials (roughness, metallic, opacity)
- **STL** - Standard format for FDM/SLA printers
- **3MF** - Modern format with color support for multi-material printers

## Prerequisites

- **NVIDIA GPU with 24GB+ VRAM** (RTX 3090, RTX 4090, L4, A100, or similar)
- **Docker** with NVIDIA Container Toolkit
- **API Keys:**
  - Replicate API token (for image generation)
  - HuggingFace token (optional, for faster model downloads)

## Quick Start

### 1. Create `.env` file

```bash
REPLICATE_API_TOKEN=r8_your_token_here
HF_TOKEN=hf_your_token_here  # Optional but recommended
```

Get tokens from:
- Replicate: https://replicate.com/account/api-tokens
- HuggingFace: https://huggingface.co/settings/tokens

### 2. Build the Docker image

```bash
# Linux/Mac
./build.sh

# Windows (in WSL or Git Bash)
bash build.sh
```

Build takes ~45-60 minutes and creates a ~50-60GB image.

### 3. Run the container

```bash
docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer
```

Open http://localhost:8080 in your browser.

---

## Local Development (RTX 3090/4090)

For development on a local workstation with an RTX 3090 or 4090:

### Prerequisites

- Docker Desktop with WSL 2 backend (Windows) or Docker (Linux)
- NVIDIA GPU drivers (535+ recommended)
- NVIDIA Container Toolkit

### Install NVIDIA Container Toolkit (Linux)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify GPU access

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Google Cloud Compute Engine Deployment

### 1. Create the VM with GPU

**Important:** Trellis 2 requires a GPU with at least 24GB VRAM. The NVIDIA L4 (24GB) is the most cost-effective option.

**Option A: Use the automated script (Windows)**

```cmd
create-gpu-vm.bat
```

This script tries multiple zones to find available GPU capacity.

**Option B: Manual creation**

```bash
# Try different zones if one is exhausted
gcloud compute instances create text-to-3d-demo \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-c \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --boot-disk-size=200GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE
```

Alternative GPU options:
- `nvidia-l4` (24GB) - Best price/performance, g2-standard-8
- `nvidia-tesla-a100` (40GB) - More powerful, a2-highgpu-1g

### 2. Open firewall for port 8080

```bash
gcloud compute firewall-rules create allow-8080 \
  --project=YOUR_PROJECT_ID \
  --allow=tcp:8080 \
  --target-tags=http-server \
  --description="Allow port 8080 for text-to-3d-printer"

gcloud compute instances add-tags text-to-3d-demo \
  --project=YOUR_PROJECT_ID \
  --tags=http-server \
  --zone=us-central1-c
```

### 3. SSH into the VM

```bash
gcloud compute ssh text-to-3d-demo --project=YOUR_PROJECT_ID --zone=us-central1-c
```

### 4. Install Docker and NVIDIA runtime

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Log out and back in for docker group to take effect
exit
```

### 5. Copy project files to VM

From your **local machine**:

```bash
# Windows
gcloud compute scp --recurse "C:\path\to\text-to-3d-printer" text-to-3d-demo:~ --project=YOUR_PROJECT_ID --zone=us-central1-c

# Linux/Mac
gcloud compute scp --recurse ./text-to-3d-printer text-to-3d-demo:~ --project=YOUR_PROJECT_ID --zone=us-central1-c
```

### 6. Build and run on VM

```bash
gcloud compute ssh text-to-3d-demo --project=YOUR_PROJECT_ID --zone=us-central1-c
cd text-to-3d-printer

# Create .env file
cat > .env << 'EOF'
REPLICATE_API_TOKEN=r8_your_token_here
HF_TOKEN=hf_your_token_here
EOF

# Build (~45-60 min first time)
./build.sh

# Run
docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer
```

### 7. Access the app

Get the external IP:
```bash
gcloud compute instances describe text-to-3d-demo \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-c \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Open `http://EXTERNAL_IP:8080` in your browser.

### 8. Stop VM when not in use

```bash
gcloud compute instances stop text-to-3d-demo \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-c
```

---

## Pushing app.py Updates (Hot Reload)

After the initial build, you can update `app.py` without rebuilding the entire image:

### One-time setup on VM

```bash
gcloud compute ssh text-to-3d-demo --project=YOUR_PROJECT_ID --zone=us-central1-c
cd ~/text-to-3d-printer

cat > run.sh << 'EOF'
#!/bin/bash
echo "Stopping any running containers..."
docker stop $(docker ps -q) 2>/dev/null

echo "Starting text-to-3d-printer..."
docker run --gpus all -p 8080:8080 --env-file .env \
  -v $(pwd)/app.py:/app/app.py:ro \
  text-to-3d-printer
EOF

chmod +x run.sh
exit
```

### Update workflow

From your **local machine**:

```bash
# 1. Copy updated app.py
gcloud compute scp app.py text-to-3d-demo:~/text-to-3d-printer/app.py --project=YOUR_PROJECT_ID --zone=us-central1-c

# 2. SSH and restart
gcloud compute ssh text-to-3d-demo --project=YOUR_PROJECT_ID --zone=us-central1-c
cd ~/text-to-3d-printer
./run.sh
```

The app restarts in seconds with your changes.

---

## Project Structure

```
├── app.py                  # Main Gradio application
├── Dockerfile              # Docker build (CUDA 12.4, PyTorch 2.6, Trellis 2)
├── build.sh                # Build script
├── create-gpu-vm.bat       # GCE VM creation script (Windows)
├── .env                    # API tokens (create this yourself)
├── README.md               # This file
└── patches/                # Legacy patches (not used with Trellis 2)
```

---

## Technical Notes

### GPU Memory Requirements

- **Minimum:** 24GB VRAM (RTX 3090, L4)
- **Recommended:** 40GB+ VRAM (A100) for higher resolution generation

### Trellis 2 Output Quality

Trellis 2 generates meshes at different resolutions:
- **512³** - ~3 seconds, good for previews
- **1024³** - ~17 seconds, balanced quality
- **1536³** - ~60 seconds, highest quality

The default configuration uses moderate settings optimized for ~30-60 second generation time.

### 3D Printing Tips

1. **Full-color printing services** (Shapeways, Sculpteo): Upload the GLB file directly
2. **Multi-material FDM** (Bambu AMS, Prusa MMU): Use the 3MF file in your slicer
3. **Standard FDM**: Print STL in white, hand-paint using GLB preview as reference
4. **Scaling**: Models may be small - scale up 10-100x in your slicer

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "CUDA out of memory" | Reduce resolution or use a GPU with more VRAM |
| "No GPU detected" | Check NVIDIA drivers and Container Toolkit installation |
| Build fails at flash-attn | Safe to ignore - fallback attention will be used |
| Model download slow | Add HF_TOKEN to .env for faster downloads |
| GCE "quota exceeded" | Request GPU quota increase in Cloud Console |

---

## License

This project uses:
- [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) - MIT License
- [Nano Banana](https://replicate.com/google/nano-banana) via Replicate
- [rembg](https://github.com/danielgatis/rembg) for background removal
