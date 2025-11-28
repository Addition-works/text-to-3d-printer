# text-to-3d-printer

Generate 3D printable models from text descriptions using SAM 3D Objects and SAM 3D Body.

**Pipeline:**
1. Text prompt → Generate images (Stable Diffusion via Replicate)
2. Select best image
3. Auto-segment object/person from background
4. Choose mode: **Objects** (products, items) or **Human Body** (poses, figures)
5. Reconstruct 3D model (SAM 3D Objects or SAM 3D Body)
6. Download STL for 3D printing

## Features

- **SAM 3D Objects**: Reconstruct 3D models of products, items, and general objects
- **SAM 3D Body**: Reconstruct 3D human body meshes with accurate pose estimation
- **Automatic segmentation**: Uses rembg to extract objects from backgrounds
- **3D printing ready**: Exports to STL format for slicers like Cura/PrusaSlicer

## Prerequisites

- Docker Desktop with WSL 2 backend
- NVIDIA GPU with drivers installed (tested with RTX 3090)
- HuggingFace account with licenses accepted for:
  - [SAM 3D Objects](https://huggingface.co/facebook/sam-3d-objects)
  - [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3)

## Setup

**1. Accept the model licenses on HuggingFace:**

- Go to https://huggingface.co/facebook/sam-3d-objects and click "Agree and access repository"
- Go to https://huggingface.co/facebook/sam-3d-body-dinov3 and click "Agree and access repository"

**2. Create `.env` file in project root:**

```
HF_TOKEN=hf_your_token_here
REPLICATE_API_TOKEN=r8_your_token_here
```

Get tokens from:
- HuggingFace: https://huggingface.co/settings/tokens
- Replicate: https://replicate.com/account/api-tokens

Add a billing method and credits to Replicate to use Stable Diffusion.

**3. Build Docker image:**

This will take roughly ~45 min to build and download model weights (~12GB total).

```bash
docker build -t text-to-3d-printer .
```

**4. Run:**

```bash
docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer
```

Open http://localhost:8080

## Usage

1. **Select Mode**: Choose between "Objects" (for products/items) or "Human Body" (for poses/figures)
2. **Enter Description**: Describe what you want to 3D print
3. **Generate Images**: Click to generate AI images based on your description
4. **Select Image**: Click on the image you like best
5. **Confirm Mask**: Review the segmentation (red overlay shows what will be 3D printed)
6. **Generate 3D**: Click to run SAM 3D reconstruction
7. **Download STL**: Save the file and import into your slicer

## Technical Notes

### SAM 3D Objects
- Reconstructs general objects from masked regions
- Outputs Gaussian splats and meshes
- Best for: products, furniture, items, toys

### SAM 3D Body
- Reconstructs human body meshes with pose estimation
- Uses the Momentum Human Rig (MHR) representation
- Best for: figurines, action poses, character models
- Does NOT require a mask - it detects humans automatically

### Known Issues

See `pr_sam3d.md` for documented issues and workarounds applied in this integration.