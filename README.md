# text-to-3d-printer

Generate 3D printable models from text descriptions using SAM 3D Objects and SAM 3D Body.

**Pipeline:**
1. Text prompt → Generate images (Stable Diffusion via Replicate)
2. Select best image
3. Auto-segment object from background
4. Reconstruct 3D model (SAM 3D Objects or SAM 3D Body)
5. Download STL for 3D printing

**Modes:**
- 🎁 **Objects** - General objects, products, items (uses SAM 3D Objects)
- 🧍 **Human Body** - Full-body human poses (uses SAM 3D Body)

## Prerequisites

- NVIDIA GPU (RTX 3090 or similar recommended)
- HuggingFace account with licenses accepted:
  - [SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects)
  - [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3)

## Setup

### Option 1: Local (Docker)

**1. Install prerequisites:**
- Docker Desktop with WSL 2 backend (Windows) or Docker (Linux)
- NVIDIA GPU drivers + NVIDIA Container Toolkit

**2. Create `.env` file:**
```
HF_TOKEN=hf_your_token_here
REPLICATE_API_TOKEN=r8_your_token_here
```

Get tokens from:
- HuggingFace: https://huggingface.co/settings/tokens
- Replicate: https://replicate.com/account/api-tokens

**3. Build and run:**
```bash
docker build -t text-to-3d-printer .
docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer
```

Open http://localhost:8080

---

### Option 2: Google Cloud Compute Engine

**1. Create the VM with GPU:**
```bash
gcloud compute instances create text-to-3d-demo \
  --project=text-to-3d-printer \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --boot-disk-size=200GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE
```

**2. Open firewall for port 8080:**
```bash
gcloud compute firewall-rules create allow-8080 \
  --project=text-to-3d-printer \
  --allow=tcp:8080 \
  --target-tags=http-server \
  --description="Allow port 8080"

gcloud compute instances add-tags text-to-3d-demo \
  --project=text-to-3d-printer \
  --tags=http-server \
  --zone=us-central1-a
```

**3. SSH into the VM:**
```bash
gcloud compute ssh text-to-3d-demo --project=text-to-3d-printer --zone=us-central1-a
```

**4. Install Docker and NVIDIA runtime:**
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

**5. SSH back in and clone the project:**
```bash
gcloud compute ssh text-to-3d-demo --project=text-to-3d-printer --zone=us-central1-a

git clone https://github.com/YOUR_USERNAME/text-to-3d-printer.git
cd text-to-3d-printer
```

**6. Create `.env` file:**
```bash
cat > .env << 'EOF'
HF_TOKEN=hf_your_token_here
REPLICATE_API_TOKEN=r8_your_token_here
EOF
```

**7. Build the image (~30-45 min):**
```bash
docker build -t text-to-3d-printer .
```

**8. Run:**
```bash
docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer
```

**9. Access the app:**

Get the external IP:
```bash
gcloud compute instances describe text-to-3d-demo \
  --project=text-to-3d-printer \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Open `http://EXTERNAL_IP:8080` in your browser.

**10. Stop the VM when not in use (to save costs):**
```bash
gcloud compute instances stop text-to-3d-demo \
  --project=text-to-3d-printer \
  --zone=us-central1-a
```

---

## Project Structure

```
├── app.py                          # Main Gradio application
├── Dockerfile                      # Docker build instructions
├── .env                            # API tokens (create this yourself)
├── download_checkpoints.py         # Helper to download model weights
├── README.md                       # This file
└── patches/
    └── utils3d_depth_patch.py      # Patch for MoGe v2 compatibility
```

## Notes

- First image generation takes longer (downloads rembg model)
- First 3D reconstruction takes longer (loads SAM 3D models)
- The Docker image is ~55GB due to CUDA, PyTorch, and model checkpoints
- Build time is ~30-45 minutes