"""
Download SAM 3D Objects checkpoints from HuggingFace.
Requires HF_TOKEN environment variable or .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# Load token from .env
load_dotenv()
hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    print("ERROR: HF_TOKEN not found!")
    print("Set it in your .env file or as an environment variable.")
    exit(1)

base_dir = Path(__file__).parent

# ---------------------------------------------------------------------------
# Download SAM 3D Objects
# ---------------------------------------------------------------------------

objects_checkpoint_dir = base_dir / "checkpoints"
objects_checkpoint_dir.mkdir(exist_ok=True)

print(f"Downloading SAM 3D Objects checkpoints to: {objects_checkpoint_dir}")
print("This is ~10GB and may take 10-20 minutes...")
print()

try:
    snapshot_download(
        repo_id="facebook/sam-3d-objects",
        local_dir=str(objects_checkpoint_dir),
        token=hf_token,
    )
    print()
    print("✓ SAM 3D Objects download complete!")
    print(f"  Checkpoints saved to: {objects_checkpoint_dir}")
except Exception as e:
    print(f"ERROR: SAM 3D Objects download failed: {e}")
    print()
    print("Make sure you have:")
    print("1. Accepted the license at https://huggingface.co/facebook/sam-3d-objects")
    print("2. A valid HF_TOKEN with read access")

print()
print("=" * 60)
print("Download summary:")
print("=" * 60)
if (objects_checkpoint_dir / "checkpoints" / "pipeline.yaml").exists():
    print("  SAM 3D Objects: ✓ Ready")
else:
    print("  SAM 3D Objects: ✗ Missing or incomplete")