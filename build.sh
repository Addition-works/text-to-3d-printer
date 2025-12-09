#!/bin/bash
# build.sh - Secure Docker build script for text-to-3d-printer
#
# This script builds the Docker image using BuildKit secrets to avoid
# baking API keys into image layers.
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
echo " text-to-3d-printer - Secure Docker Build"
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

# Validate we have a token
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}ERROR: HF_TOKEN not found!${NC}"
    echo ""
    echo "Please either:"
    echo "  1. Set HF_TOKEN environment variable:"
    echo "     export HF_TOKEN=hf_your_token_here"
    echo "     ./build.sh"
    echo ""
    echo "  2. Or create a .env file with:"
    echo "     HF_TOKEN=hf_your_token_here"
    echo ""
    exit 1
fi

# Show token prefix for verification (don't show full token!)
TOKEN_PREFIX="${HF_TOKEN:0:10}"
echo -e "${GREEN}✓ Found HF_TOKEN (starts with: ${TOKEN_PREFIX}...)${NC}"
echo ""

# Ensure BuildKit is enabled
export DOCKER_BUILDKIT=1

echo "Building Docker image with BuildKit secrets..."
echo "(Your HF_TOKEN is passed securely and NOT stored in image layers)"
echo ""

# Build with secret - the token is mounted at /run/secrets/hf_token during build only
docker build \
    --secret id=hf_token,env=HF_TOKEN \
    -t text-to-3d-printer \
    .

echo ""
echo -e "${GREEN}=========================================="
echo " Build complete!"
echo "==========================================${NC}"
echo ""
echo "To run the container:"
echo "  docker run --gpus all -p 8080:8080 --env-file .env text-to-3d-printer"
echo ""
echo "Note: The .env file is needed at RUNTIME for REPLICATE_API_TOKEN,"
echo "      but it is NOT baked into the image."
