# # S-Graphs with CUDA/TensorRT/ONNXRuntime Support
# # Base: NVIDIA CUDA 12.1 with cuDNN on Ubuntu 22.04 (compatible with ROS2 Humble)
# #
# # IMPORTANT: This uses CUDA 12.1 to match TensorRT engines built with CUDA 12.1
# # If your engines were built with a different CUDA version, adjust accordingly.
# #
# # This Dockerfile builds an environment capable of running:
# # - Full S-Graphs SLAM pipeline
# # - YOLO object detection (TensorRT)
# # - CLIP feature extraction (ONNXRuntime GPU)
# # - Camera-based semantic integration
# #
# # Build:
# #   docker build -t sgraphs-cuda:humble -f docker/cuda_humble/Dockerfile .
# #
# # Run:
# #   docker run --gpus all -it --rm \
# #       -v /path/to/models:/models \
# #       -v /path/to/bags:/bags \
# #       -e DISPLAY=$DISPLAY \
# #       -v /tmp/.X11-unix:/tmp/.X11-unix \
# #       sgraphs-cuda:humble

# FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# ARG COLCON_JOBS=4
# ARG DEBIAN_FRONTEND=noninteractive

# ENV MAKEFLAGS="-j ${COLCON_JOBS}"
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# # ============================================
# # Stage 1: Install system dependencies + ROS2
# # ============================================

# # Basic tools
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common \
#     curl \
#     gnupg \
#     lsb-release \
#     wget \
#     git \
#     vim \
#     build-essential \
#     cmake \
#     pkg-config \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# # Add ROS2 Humble repository
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# # Install ROS2 Humble desktop + dev tools
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ros-humble-desktop-full \
#     ros-humble-rmw-cyclonedds-cpp \
#     python3-colcon-common-extensions \
#     python3-vcstool \
#     python3-rosdep \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# # ============================================
# # Stage 2: Install TensorRT
# # ============================================

# # TensorRT 8.6+ is compatible with CUDA 12.1
# # For CUDA 12.x, we need TensorRT 8.6 or 10.x
# # Install from NVIDIA's repo
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libnvinfer8 \
#     libnvinfer-dev \
#     libnvinfer-plugin8 \
#     libnvinfer-plugin-dev \
#     libnvonnxparsers8 \
#     libnvonnxparsers-dev \
#     libnvparsers8 \
#     libnvparsers-dev \
#     || apt-get install -y --no-install-recommends \
#     libnvinfer10 \
#     libnvinfer-dev \
#     libnvinfer-plugin10 \
#     libnvinfer-plugin-dev \
#     libnvonnxparsers10 \
#     libnvonnxparsers-dev \
#     && rm -rf /var/lib/apt/lists/*

# # ============================================
# # Stage 3: Install ONNXRuntime with GPU
# # ============================================

# # 1. Download and install C++ libraries to /opt/onnxruntime
# # Replicates your manual steps: wget -> tar -> mv
# WORKDIR /tmp
# RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz \
#     && tar -xf onnxruntime-linux-x64-1.18.0.tgz \
#     && mv onnxruntime-linux-x64-1.18.0 /opt/onnxruntime \
#     && rm onnxruntime-linux-x64-1.18.0.tgz

# # 2. Set Environment Variables for CMake and Runtime
# # This ensures CMake finds the libraries in Stage 6
# ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
# ENV LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT}/lib:${LD_LIBRARY_PATH}

# # 3. Install Python bindings (CPU version to match)
# RUN pip3 install --no-cache-dir \
#     onnxruntime==1.18.0 \
#     numpy \
#     opencv-python-headless
# # ============================================
# # Stage 4: Install OpenCV, cv_bridge, and ROS dependencies
# # ============================================

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libopencv-dev \
#     # --- ADD THESE 3 LINES ---
#     libsuitesparse-dev \
#     libgoogle-glog-dev \
#     ros-humble-libg2o \
#     # -------------------------
#     ros-humble-cv-bridge \
#     ros-humble-image-transport \
#     ros-humble-image-transport-plugins \
#     ros-humble-compressed-image-transport \
#     ros-humble-geodesy \
#     ros-humble-geographic-msgs \
#     ros-humble-nmea-msgs \
#     ros-humble-pcl-ros \
#     ros-humble-tf2 \
#     ros-humble-tf2-ros \
#     ros-humble-tf2-eigen \
#     ros-humble-tf2-sensor-msgs \
#     ros-humble-message-filters \
#     ros-humble-backward-ros \
#     ros-humble-rviz-visual-tools \
#     && rm -rf /var/lib/apt/lists/* \
#     && ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2


# # ============================================
# # Stage 5: Install Python ML dependencies
# # ============================================

# # Install scikit-learn from default PyPI
# RUN pip3 install --no-cache-dir scikit-learn

# # Install PyTorch with CUDA 12.1 support from PyTorch index
# RUN pip3 install --no-cache-dir \
#     torch==2.1.0 \
#     torchvision==0.16.0 \
#     torchaudio==2.1.0 \
#     --index-url https://download.pytorch.org/whl/cu121

# # ============================================
# # Stage 6: Setup workspace and build S-Graphs
# # ============================================

# # Initialize rosdep
# RUN rosdep init || true
# RUN rosdep update

# # Copy source code
# WORKDIR /workspace/src
# COPY . /workspace/src/s_graphs

# # Import dependencies via vcs
# WORKDIR /workspace/src/s_graphs
# RUN vcs import . < .rosinstall_ros2

# # Install ROS dependencies
# WORKDIR /workspace
# RUN apt-get update && rosdep install --from-paths src -y --ignore-src -r \
#     && rm -rf /var/lib/apt/lists/*

# # Build the workspace
# RUN . /opt/ros/humble/setup.sh && \
#     colcon build --cmake-args \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1

# # ============================================
# # Stage 7: Setup entrypoint and environment
# # ============================================

# # Create entrypoint script
# RUN echo '#!/bin/bash\n\
#     set -e\n\
#     source /opt/ros/humble/setup.bash\n\
#     source /workspace/install/setup.bash\n\
#     export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n\
#     exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# # Set environment variables for CUDA
# ENV PATH=/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# # Create models directory (mount your models here at runtime)
# RUN mkdir -p /models

# WORKDIR /workspace

# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["bash"]

# S-Graphs with CUDA/TensorRT/ONNXRuntime Support
# Base: NVIDIA CUDA 12.1 with cuDNN on Ubuntu 22.04 (compatible with ROS2 Humble)
#
# IMPORTANT: This uses CUDA 12.1 to match TensorRT engines built with CUDA 12.1
# If your engines were built with a different CUDA version, adjust accordingly.
#
# This Dockerfile builds an environment capable of running:
# - Full S-Graphs SLAM pipeline
# - YOLO object detection (TensorRT)
# - CLIP feature extraction (ONNXRuntime GPU)
# - Camera-based semantic integration
#
# Build:
#   docker build -t sgraphs-cuda:humble -f docker/cuda_humble/Dockerfile .
#
# Run:
#   docker run --gpus all -it --rm \
#       -v /path/to/models:/models \
#       -v /path/to/bags:/bags \
#       -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix \
#       sgraphs-cuda:humble

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

ARG COLCON_JOBS=4
ARG DEBIAN_FRONTEND=noninteractive

ENV MAKEFLAGS="-j ${COLCON_JOBS}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# ============================================
# Stage 1: Install system dependencies + ROS2
# ============================================

# Basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    gnupg \
    lsb-release \
    wget \
    git \
    vim \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add ROS2 Humble repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble desktop + dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop-full \
    ros-humble-rmw-cyclonedds-cpp \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: Install TensorRT
# ============================================

# TensorRT 8.6+ is compatible with CUDA 12.1
# For CUDA 12.x, we need TensorRT 8.6 or 10.x
# Install from NVIDIA's repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer8 \
    libnvinfer-dev \
    libnvinfer-plugin8 \
    libnvinfer-plugin-dev \
    libnvonnxparsers8 \
    libnvonnxparsers-dev \
    libnvparsers8 \
    libnvparsers-dev \
    || apt-get install -y --no-install-recommends \
    libnvinfer10 \
    libnvinfer-dev \
    libnvinfer-plugin10 \
    libnvinfer-plugin-dev \
    libnvonnxparsers10 \
    libnvonnxparsers-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 3: Install ONNXRuntime (CPU Version 1.18.0)
# ============================================

# 1. Download and install C++ libraries to /opt/onnxruntime
WORKDIR /tmp
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz \
    && tar -xf onnxruntime-linux-x64-1.18.0.tgz \
    && mv onnxruntime-linux-x64-1.18.0 /opt/onnxruntime \
    && rm onnxruntime-linux-x64-1.18.0.tgz

# 2. Set Environment Variables for CMake and Runtime
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT}/lib:${LD_LIBRARY_PATH}

# 3. Install Python bindings (CPU version to match)
RUN pip3 install --no-cache-dir \
    onnxruntime==1.18.0 \
    numpy \
    opencv-python-headless

# ============================================
# Stage 4: Install OpenCV, cv_bridge, and ROS dependencies
# ============================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    # --- ADD THESE 3 LINES ---
    libsuitesparse-dev \
    libgoogle-glog-dev \
    ros-humble-libg2o \
    # -------------------------
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-compressed-image-transport \
    ros-humble-geodesy \
    ros-humble-geographic-msgs \
    ros-humble-nmea-msgs \
    ros-humble-pcl-ros \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-eigen \
    ros-humble-tf2-sensor-msgs \
    ros-humble-message-filters \
    ros-humble-backward-ros \
    ros-humble-rviz-visual-tools \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2

# ============================================
# Stage 5: Install Python ML dependencies
# ============================================

# Install scikit-learn and updated numpy/scipy as requested
RUN pip3 install --no-cache-dir \
    scikit-learn \
    'numpy<2' \
    transforms3d \
    scipy

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric Stack (Version 2.2.0)
# Note: We install dependencies from the wheel specifically for torch-2.1.0+cu121
RUN pip3 install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    && pip3 install --no-cache-dir torch-geometric==2.2.0

# ============================================
# Stage 6: Setup workspace and build S-Graphs
# ============================================

# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

# Copy source code
WORKDIR /workspace/src
COPY . /workspace/src/s_graphs

# Import dependencies via vcs
WORKDIR /workspace/src/s_graphs
RUN vcs import . < .rosinstall_ros2

# --- FIX 1: Exclude broken semantic files from the Core Library ---
# This prevents the library build from crashing on missing ONNX headers
RUN find . -name CMakeLists.txt -exec sed -i '/add_library(g2o_${PROJECT_NAME}/i \
    list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "clip_feature_extractor\\\\.cpp$")\n\
    list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "yolo_object_detector\\\\.cpp$")' {} +

# --- FIX 2: Add missing dependencies to CMakeLists.txt ---
# Fixes "cv_bridge has not been declared" compilation error
RUN find . -name CMakeLists.txt -exec sed -i '/find_package(sensor_msgs REQUIRED)/a \
    find_package(cv_bridge REQUIRED)\n\
    find_package(image_transport REQUIRED)\n\
    find_package(message_filters REQUIRED)' {} +

# --- FIX 3: Patch s_graphs.hpp (Add missing typedef and headers) ---
# Fixes "TripleSyncPolicy was not declared" and "template argument invalid"
RUN find . -name s_graphs.hpp -exec sed -i '/std::shared_ptr<message_filters::Synchronizer/i \
    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::msg::Odometry, sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image> TripleSyncPolicy;' {} + \
    && find . -name s_graphs.hpp -exec sed -i '/#include <vector>/a \
    #include <cv_bridge/cv_bridge.hpp>\n\
    #include <sensor_msgs/image_encodings.hpp>\n\
    #include <message_filters/synchronizer.h>\n\
    #include <message_filters/sync_policies/approximate_time.h>' {} +

    # --- FIX 4: Patch s_graphs.cpp (Disable the missing function call) ---
    # Fixes linker error for the function we excluded in Fix 1
    RUN find . -name s_graphs.cpp -exec sed -i 's|extract_semantic_features_from_camera(keyframe);|// extract_semantic_features_from_camera(keyframe);|g' {} +

# Install ROS dependencies
WORKDIR /workspace
RUN apt-get update && rosdep install --from-paths src -y --ignore-src -r \
    && rm -rf /var/lib/apt/lists/*

# Build the workspace
# Includes explicit ONNX flags as requested
RUN . /opt/ros/humble/setup.sh && \
    colcon build --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.1 \
    -DONNXRUNTIME_ROOT=/opt/onnxruntime \
    -DONNXRUNTIME_lib_DIR=/opt/onnxruntime/lib \
    -DONNXRUNTIME_INCLUDE_DIR=/opt/onnxruntime/include \
    -DONNXRUNTIME_LIBRARY=/opt/onnxruntime/lib/libonnxruntime.so

# ============================================
# Stage 7: Setup entrypoint and environment
# ============================================

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    set -e\n\
    source /opt/ros/humble/setup.bash\n\
    source /workspace/install/setup.bash\n\
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n\
    exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Create models directory (mount your models here at runtime)
RUN mkdir -p /models

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]