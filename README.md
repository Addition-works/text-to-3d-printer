# text-to-3d-printer

Generate 3D printable models from text descriptions using SAM 3D Objects.

**Pipeline:**
1. Text prompt → Generate images (Stable Diffusion via Replicate)
2. Select best image
3. Auto-segment object from background
4. Reconstruct 3D model (SAM 3D Objects)
5. Download STL for 3D printing

**Modes:**
- 🎁 **Objects** - General objects, products, items (uses SAM 3D Objects)

## Prerequisites

- NVIDIA GPU (RTX 3090 or similar recommended)
- HuggingFace account with license accepted:
  - [SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects)

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

> **Note:** GPU availability varies by zone. If a zone is exhausted, try another (us-west1-b, us-east4-c, europe-west4-a, etc.)

```bash
gcloud compute instances create text-to-3d-demo ^
  --project=text-to-3d-printer ^
  --zone=us-west1-a ^
  --machine-type=g2-standard-8 ^
  --accelerator=type=nvidia-l4,count=1 ^
  --boot-disk-size=200GB ^
  --image-family=ubuntu-2204-lts ^
  --image-project=ubuntu-os-cloud ^
  --maintenance-policy=TERMINATE
```

**2. Open firewall for port 8080:**
```bash
gcloud compute firewall-rules create allow-8080 ^
  --project=text-to-3d-printer ^
  --allow=tcp:8080 ^
  --target-tags=http-server ^
  --description="Allow port 8080"

gcloud compute instances add-tags text-to-3d-demo ^
  --project=text-to-3d-printer ^
  --tags=http-server ^
  --zone=us-west1-a
```

**3. SSH into the VM:**
```bash
gcloud compute ssh text-to-3d-demo --project=text-to-3d-printer --zone=us-west1-a
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

**5. Copy project files from your local machine:**

Run this from your **local terminal** (not SSH):

```bash
gcloud compute scp --recurse C:\Users\machine314\Desktop\text-to-3d-printer text-to-3d-demo: --project=text-to-3d-printer --zone=us-west1-a
```

Replace `C:\Users\machine314\Desktop\text-to-3d-printer` with the actual path to your project folder.

**6. SSH back in:**
```bash
gcloud compute ssh text-to-3d-demo --project=text-to-3d-printer --zone=us-west1-a
cd text-to-3d-printer
```

**7. Create `.env` file:**
```bash
cat > .env << 'EOF'
HF_TOKEN=hf_your_token_here
REPLICATE_API_TOKEN=r8_your_token_here
EOF
```

**8. Build the image (~55 min):**
```bash
docker build -t text-to-3d-printer .
```

**9. Run:**
```bash
docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer
```

**10. Access the app:**

Get the external IP:
```bash
gcloud compute instances describe text-to-3d-demo ^
  --project=text-to-3d-printer ^
  --zone=us-west1-a ^
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Open `http://EXTERNAL_IP:8080` in your browser.

**11. Stop the VM when not in use (to save costs):**
```bash
gcloud compute instances stop text-to-3d-demo ^
  --project=text-to-3d-printer ^
  --zone=us-west1-a
```

---

## Pushing app.py Revisions to GCE

When you make changes to `app.py`, you don't need to rebuild the Docker image (which takes ~55 min). Instead, use a volume mount to override the file inside the container.

### One-Time Setup

SSH into the VM and create a helper script:

```bash
gcloud compute ssh text-to-3d-demo --project=text-to-3d-printer --zone=us-west1-a
```

```bash
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
```

Then exit:
```bash
exit
```

### Update Workflow

From your **local terminal**, run these commands whenever you change `app.py`:

**1. Copy the updated file to the VM:**
```bash
gcloud compute scp C:\Users\machine314\Desktop\text-to-3d-printer\app.py text-to-3d-demo:app.py --project=text-to-3d-printer --zone=us-west1-a
```

> **Note:** Windows doesn't expand `~`, so we copy to the home directory without a path, then move it after SSH.

**2. SSH in, move the file, and restart:**
```bash
gcloud compute ssh text-to-3d-demo --project=text-to-3d-printer --zone=us-west1-a
```

```bash
mv ~/app.py ~/text-to-3d-printer/app.py
cd ~/text-to-3d-printer
./run.sh
```

The app will restart with your changes in seconds (no rebuild needed).

### How It Works

The `-v $(pwd)/app.py:/app/app.py:ro` flag mounts your local `app.py` over the one baked into the Docker image:
- The container still uses all dependencies and model checkpoints from the image
- Only `app.py` is read from the VM's filesystem
- `:ro` means read-only (the container can't modify your file)

---

## Project Structure

```
├── app.py                          # Main Gradio application
├── Dockerfile                      # Docker build instructions
├── .env                            # API tokens (create this yourself)
├── download_checkpoints.py         # Helper to download model weights
├── README.md                       # This file
├── run.sh                          # Helper script for quick restarts (created on VM)
└── patches/
    └── utils3d_depth_patch.py      # Patch for MoGe v2 compatibility
```

## Notes

- First image generation takes longer (downloads rembg model)
- First 3D reconstruction takes longer (loads SAM 3D models)
- The Docker image is ~55GB due to CUDA, PyTorch, and model checkpoints
- Build time is ~30-45 minutes