#!/bin/bash
# Build script for S-Graphs CUDA Docker image
# Usage: ./build_cuda_docker.sh [--no-cache]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="sgraphs-cuda"
IMAGE_TAG="humble"

echo "============================================"
echo "Building S-Graphs CUDA Docker Image"
echo "============================================"
echo "Repository root: ${REPO_ROOT}"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Check for --no-cache flag
CACHE_FLAG=""
if [[ "$1" == "--no-cache" ]]; then
    CACHE_FLAG="--no-cache"
    echo "Building without cache..."
fi

# Build the image
cd "${REPO_ROOT}"
docker build \
    ${CACHE_FLAG} \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -f docker/cuda_humble/Dockerfile \
    .

echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo ""
echo "To run the container:"
echo ""
echo "  # Basic run with GPU:"
echo "  docker run --gpus all -it --rm ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "  # With display (for RViz):"
echo "  xhost +local:docker"
echo "  docker run --gpus all -it --rm \\"
echo "      -e DISPLAY=\$DISPLAY \\"
echo "      -v /tmp/.X11-unix:/tmp/.X11-unix \\"
echo "      ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "  # With model files and ROS bags:"
echo "  docker run --gpus all -it --rm \\"
echo "      -v /path/to/models:/models \\"
echo "      -v /path/to/bags:/bags \\"
echo "      ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
