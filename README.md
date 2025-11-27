# text-to-3d-printer

Generate 3D printable models from text descriptions using SAM 3D Objects.

**Pipeline:**
1. Text prompt → Generate 4 images (Stable Diffusion via Replicate)
2. Select best image
3. Auto-segment object from background
4. Reconstruct 3D model (SAM 3D Objects)
5. Download STL for 3D printing

## Prerequisites

- Docker Desktop with WSL 2 backend
- NVIDIA GPU with drivers installed
- HuggingFace account with SAM 3D Objects license accepted

## Setup

**1. Accept the SAM 3D Objects license:**

Go to https://huggingface.co/facebook/sam-3d-objects and click "Agree and access repository"

**2. Create `.env` file in project root:**

```
HF_TOKEN=hf_your_token_here
REPLICATE_API_TOKEN=r8_your_token_here
```

Get tokens from:
- HuggingFace: https://huggingface.co/settings/tokens
- Replicate: https://replicate.com/account/api-tokens

**3. Build Docker image:**

```bash
docker build -t text-to-3d-printer .
```

This takes 20-40 minutes (downloads ~10GB of model weights).

**4. Run:**

```bash
docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer
```

Open http://localhost:8080