#!/bin/bash
# Run script for S-Graphs CUDA Docker container
# Usage: ./run_cuda_docker.sh [--rviz] [--models /path/to/models] [--bags /path/to/bags]

set -e

IMAGE_NAME="sgraphs-cuda"
IMAGE_TAG="humble"
CONTAINER_NAME="sgraphs_CUDA_container"


# Default paths (modify these for your setup)
MODELS_PATH="${MODELS_PATH:-$HOME/models}"
BAGS_PATH="${BAGS_PATH:-$HOME/bags}"

# Parse arguments
ENABLE_DISPLAY=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rviz|--pip install 'numpy<2'

pip install --upgrade transforms3d

pip install --upgrade scipy

pip3 uninstall -y torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip3 install torch-geometric==2.0.4

pip3 uninstall -y torch-geometric

pip3 install torch-geometric==2.2.0display)
            ENABLE_DISPLAY=true
            shift
            ;;
        --models)
            MODELS_PATH="$2"
            shift 2
            ;;
        --bags)
            BAGS_PATH="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# Build docker run command
DOCKER_CMD="docker run --name ${CONTAINER_NAME} --gpus all -it"

# Add network for ROS2 communication
DOCKER_CMD="${DOCKER_CMD} --network host"

# Add display if requested
if [ "$ENABLE_DISPLAY" = true ]; then
    echo "Enabling display for RViz/GUI..."
    xhost +local:docker 2>/dev/null || true
    DOCKER_CMD="${DOCKER_CMD} -e DISPLAY=${DISPLAY}"
    DOCKER_CMD="${DOCKER_CMD} -v /tmp/.X11-unix:/tmp/.X11-unix"
    DOCKER_CMD="${DOCKER_CMD} -e QT_X11_NO_MITSHM=1"
fi

# Add volume mounts if paths exist
if [ -d "${MODELS_PATH}" ]; then
    echo "Mounting models from: ${MODELS_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v ${MODELS_PATH}:/models:ro"
else
    echo "Warning: Models path not found: ${MODELS_PATH}"
    echo "         Create directory or use --models /path/to/models"
fi

if [ -d "${BAGS_PATH}" ]; then
    echo "Mounting bags from: ${BAGS_PATH}"
    DOCKER_CMD="${DOCKER_CMD} -v ${BAGS_PATH}:/bags:ro"
fi

# Add image name
DOCKER_CMD="${DOCKER_CMD} ${IMAGE_NAME}:${IMAGE_TAG}"

# Add any extra arguments
DOCKER_CMD="${DOCKER_CMD} ${EXTRA_ARGS}"

echo ""
echo "Running: ${DOCKER_CMD}"
echo ""

# Execute
eval ${DOCKER_CMD}
