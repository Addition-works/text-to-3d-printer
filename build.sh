#!/bin/bash
# build.sh - Docker build script for text-to-3d-printer (Trellis 2 version)
#
# This script builds the Docker image for the Trellis 2-powered 3D model generator.
# The HF_TOKEN is used during build to download model weights from HuggingFace.
#
# Usage:
#   ./build.sh                    # Uses HF_TOKEN from environment or .env file
#   HF_TOKEN=hf_xxx ./build.sh    # Pass token directly
#
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo " text-to-3d-printer - Trellis 2 Build"
echo "=========================================="
echo ""

# Try to get HF_TOKEN from environment, or load from .env
if [ -z "$HF_TOKEN" ]; then
    if [ -f ".env" ]; then
        echo -e "${YELLOW}Loading HF_TOKEN from .env file...${NC}"
        # Extract HF_TOKEN from .env (handles various formats)
        HF_TOKEN=$(grep -E "^HF_TOKEN=" .env | cut -d'=' -f2- | tr -d '"' | tr -d "'" | tr -d '\r')
    fi
fi

# Validate we have a token (optional for Trellis 2 public model, but recommended)
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}WARNING: HF_TOKEN not found!${NC}"
    echo ""
    echo "The Trellis 2 model (microsoft/TRELLIS.2-4B) is public, but"
    echo "having an HF_TOKEN may help avoid rate limits during download."
    echo ""
    echo "To set a token, either:"
    echo "  1. Set HF_TOKEN environment variable:"
    echo "     export HF_TOKEN=hf_your_token_here"
    echo "     ./build.sh"
    echo ""
    echo "  2. Or create a .env file with:"
    echo "     HF_TOKEN=hf_your_token_here"
    echo ""
    echo "Continuing without token..."
    echo ""
else
    # Show token prefix for verification (don't show full token!)
    TOKEN_PREFIX="${HF_TOKEN:0:10}"
    echo -e "${GREEN}Found HF_TOKEN (starts with: ${TOKEN_PREFIX}...)${NC}"
    echo ""
fi

# Ensure BuildKit is enabled for better caching
export DOCKER_BUILDKIT=1

echo "Building Docker image..."
echo "This will:"
echo "  - Install CUDA 12.4, PyTorch 2.6.0, and Trellis 2 dependencies"
echo "  - Clone and install microsoft/TRELLIS.2"
echo "  - Download the TRELLIS.2-4B model weights (~10GB)"
echo ""
echo "Estimated build time: 45-60 minutes (first build)"
echo "Estimated image size: ~50-60 GB"
echo ""

# Build the image
docker build \
    -t text-to-3d-printer \
    .

echo ""
echo -e "${GREEN}=========================================="
echo " Build complete!"
echo "==========================================${NC}"
echo ""
echo "To run the container locally:"
echo "  docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer"
echo ""
echo "Required environment variables in .env:"
echo "  REPLICATE_API_TOKEN=r8_your_token_here  (for image generation)"
echo ""
echo "Optional environment variables:"
echo "  HF_TOKEN=hf_your_token_here  (for HuggingFace access)"
echo ""
echo "Open http://localhost:8080 in your browser."
echo ""
