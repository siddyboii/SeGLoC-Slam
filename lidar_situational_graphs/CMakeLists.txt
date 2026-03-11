# SPDX-License-Identifier: BSD-2-Clause
cmake_minimum_required(VERSION 3.5)
project(lidar_situational_graphs)

# Can we use C++17 in indigo?
add_definitions(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# pcl 1.7 causes a segfault when it is built with debug mode
if (NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(geodesy REQUIRED)
find_package(nmea_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(fast_gicp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(backward_ros REQUIRED)
find_package(rviz_visual_tools REQUIRED)
find_package(situational_graphs_msgs REQUIRED)
find_package(situational_graphs_reasoning_msgs REQUIRED)

# Semantic test harness dependencies
# Rationale: cv_bridge is required for converting ROS Image messages to OpenCV Mat.
# image_transport provides optimized image subscription. CUDA and TensorRT are needed
# for YOLO inference. ONNXRuntime is needed for CLIP inference.
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# ── CUDA is OPTIONAL ──
# On CPU-only systems, YOLO runs via ONNXRuntime instead of TensorRT
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(STATUS "CUDA NOT found — building in CPU-only mode (ONNX YOLO)")
endif()

# Find TensorRT (optional — only available with CUDA)
if(CUDA_FOUND)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${CUDA_TOOLKIT_ROOT_DIR} /usr/include/x86_64-linux-gnu /usr/local/include
        PATH_SUFFIXES include)
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS ${CUDA_TOOLKIT_ROOT_DIR} /usr/lib/x86_64-linux-gnu /usr/local/lib
        PATH_SUFFIXES lib lib64)
    find_library(TENSORRT_ONNXPARSER nvonnxparser
        HINTS ${CUDA_TOOLKIT_ROOT_DIR} /usr/lib/x86_64-linux-gnu /usr/local/lib
        PATH_SUFFIXES lib lib64)
endif()

# Find ONNXRuntime — used for CLIP always, and YOLO on CPU-only builds
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS /usr/include /usr/local/include ${ONNXRUNTIME_ROOT}/include
    PATH_SUFFIXES onnxruntime)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS /usr/lib /usr/local/lib ${ONNXRUNTIME_ROOT}/lib)

# Determine build mode: TensorRT (GPU) vs ONNX-only (CPU)
if(TENSORRT_LIBRARY AND CUDA_FOUND)
    set(USE_TENSORRT_YOLO ON)
    message(STATUS "YOLO backend: TensorRT (GPU)")
else()
    set(USE_TENSORRT_YOLO OFF)
    message(STATUS "YOLO backend: ONNXRuntime (CPU)")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "ONNXRuntime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNXRuntime library: ${ONNXRUNTIME_LIBRARY}")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

option(USE_RGB_CLOUD "Use RGB point cloud" OFF)
if(USE_RGB_CLOUD)
    add_definitions(-DUSE_RGB_CLOUD)
endif()

message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})

find_package(G2O REQUIRED)
include_directories(SYSTEM ${G2O_INCLUDE_DIR} ${G2O_INCLUDE_DIRS})
link_directories(${G2O_LIBRARY_DIRS})
# link_libraries(${G2O_LIBRARIES})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Check ROS2 distribution and set appropriate header extension flag
if(DEFINED ENV{ROS_DISTRO})
    set(ROS_DISTRO $ENV{ROS_DISTRO})
    if(${ROS_DISTRO} STREQUAL "jazzy" OR ${ROS_DISTRO} STREQUAL "rolling")
        add_definitions(-DUSE_HPP_HEADERS)
        message(STATUS "ROS2 ${ROS_DISTRO} detected - using .hpp headers")
    else()
        message(STATUS "ROS${ROS_VERSION} ${ROS_DISTRO} detected - using .h headers")
    endif()
else()
    message(WARNING "ROS_DISTRO environment variable not set, using .h headers as default")
endif()

find_library(VGICP_CUDA_FOUND NAMES fast_vgicp_cuda)
message(STATUS "VGICP_CUDA_FOUND:" ${VGICP_CUDA_FOUND})
if(VGICP_CUDA_FOUND)
  add_definitions(-DUSE_VGICP_CUDA)
endif()

FIND_PACKAGE(Cholmod)
include_directories(${CHOLMOD_INCLUDE_DIR})

###########
## Build ##
###########
include_directories(include)
include_directories(
  ${PCL_INCLUDE_DIRS}
)

set(DEPENDENCIES
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    ndt_omp
    fast_gicp
    tf2
    tf2_ros
    tf2_eigen
    tf2_sensor_msgs
    pcl_ros
    visualization_msgs
    message_filters
    geographic_msgs
    nmea_msgs
    geodesy
    rviz_visual_tools
    situational_graphs_msgs
    situational_graphs_reasoning_msgs
    cv_bridge
    image_transport)

SET(G2O_COMMON_LIBS
  ${PCL_LIBRARIES}
  ${G2O_TYPES_DATA}
  ${G2O_CORE_LIBRARY}
  ${G2O_STUFF_LIBRARY}
  ${G2O_SOLVER_PCG}
  ${G2O_SOLVER_CSPARSE}   # be aware of that CSPARSE is released under LGPL
  ${G2O_SOLVER_CHOLMOD}   # be aware of that cholmod is released under GPL
  ${G2O_TYPES_SLAM3D}
  ${G2O_TYPES_SLAM3D_ADDONS})

FILE(GLOB_RECURSE G2O_S_GRAPHS_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/g2o/*.cpp
)

FILE(GLOB_RECURSE S_GRAPHS_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/s_graphs/*.cpp
)

# Conditionally include semantic feature files based on available backends
if(USE_TENSORRT_YOLO)
  # GPU mode: use TensorRT YOLO, exclude ONNX YOLO
  list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "yolo_onnx_detector\\.cpp$")
  if(NOT ONNXRUNTIME_LIBRARY)
    list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "clip_feature_extractor\\.cpp$")
  endif()
else()
  # CPU mode: use ONNX YOLO, exclude TensorRT YOLO
  list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "yolo_object_detector\\.cpp$")
  if(NOT ONNXRUNTIME_LIBRARY)
    list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "clip_feature_extractor\\.cpp$")
    list(FILTER S_GRAPHS_SOURCE_FILES EXCLUDE REGEX "yolo_onnx_detector\\.cpp$")
  endif()
endif()

add_library(g2o_${PROJECT_NAME}
  ${G2O_S_GRAPHS_SOURCE_FILES}
  )
target_link_libraries(g2o_${PROJECT_NAME}
                    ${G2O_COMMON_LIBS}
                    ${PCL_LIBRARIES}
                    )

add_library(${PROJECT_NAME}_core_lib
    ${S_GRAPHS_SOURCE_FILES}
)
target_link_libraries(${PROJECT_NAME}_core_lib
                      g2o_${PROJECT_NAME}
                      ${G2O_COMMON_LIBS}
                      ${PCL_LIBRARIES}
                      )

# Link semantic dependencies based on build mode
if(USE_TENSORRT_YOLO AND ONNXRUNTIME_LIBRARY)
  # Full GPU mode: TensorRT YOLO + ONNX CLIP
  target_compile_definitions(${PROJECT_NAME}_core_lib PRIVATE USE_TENSORRT_YOLO)
  target_include_directories(${PROJECT_NAME}_core_lib PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(${PROJECT_NAME}_core_lib
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER}
    ${CUDA_LIBRARIES}
    ${ONNXRUNTIME_LIBRARY}
    ${OpenCV_LIBRARIES}
  )
elseif(ONNXRUNTIME_LIBRARY)
  # CPU mode: ONNX YOLO + ONNX CLIP
  target_include_directories(${PROJECT_NAME}_core_lib PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(${PROJECT_NAME}_core_lib
    ${ONNXRUNTIME_LIBRARY}
    ${OpenCV_LIBRARIES}
  )
endif()
# Note: semantic_test_node is compiled separately and only when TensorRT + ONNXRuntime are available
# The core library does NOT include CLIP/YOLO code
target_include_directories(${PROJECT_NAME}_core_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME}_core_lib ${DEPENDENCIES})

ament_export_targets(export_s_graph_lib HAS_LIBRARY_TARGET)
ament_export_dependencies(${DEPENDENCIES})
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_core_lib)


add_executable(s_graphs_node
  apps/s_graphs_node.cpp
)
ament_target_dependencies(s_graphs_node ${DEPENDENCIES})

# Conditionally add include dirs and link libraries for s_graphs_node
if(USE_TENSORRT_YOLO)
  target_compile_definitions(s_graphs_node PRIVATE USE_TENSORRT_YOLO)
  target_include_directories(s_graphs_node PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(s_graphs_node
    ${PROJECT_NAME}_core_lib
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER}
    ${ONNXRUNTIME_LIBRARY}
    ${OpenCV_LIBRARIES}
  )
elseif(ONNXRUNTIME_LIBRARY)
  target_include_directories(s_graphs_node PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(s_graphs_node
    ${PROJECT_NAME}_core_lib
    ${ONNXRUNTIME_LIBRARY}
    ${OpenCV_LIBRARIES}
  )
else()
  target_link_libraries(s_graphs_node
    ${PROJECT_NAME}_core_lib
  )
endif()

# nodes
add_executable(s_graphs_prefiltering_node apps/prefiltering_node.cpp)
ament_target_dependencies(s_graphs_prefiltering_node ${DEPENDENCIES})
target_link_libraries(s_graphs_prefiltering_node
  ${PCL_LIBRARIES}
)

add_executable(s_graphs_room_segmentation_node
  apps/room_segmentation_node.cpp
)
ament_target_dependencies(s_graphs_room_segmentation_node ${DEPENDENCIES})
target_link_libraries(s_graphs_room_segmentation_node
  ${PROJECT_NAME}_core_lib
  ${PCL_LIBRARIES}
)

add_executable(s_graphs_floor_plan_node
  apps/floor_plan_node.cpp
)
ament_target_dependencies(s_graphs_floor_plan_node ${DEPENDENCIES})
target_link_libraries(s_graphs_floor_plan_node
  ${PROJECT_NAME}_core_lib
  ${PCL_LIBRARIES}
)

add_executable(s_graphs_scan_matching_odometry_node apps/scan_matching_odometry_node.cpp)
ament_target_dependencies(s_graphs_scan_matching_odometry_node ${DEPENDENCIES})
target_link_libraries(s_graphs_scan_matching_odometry_node
  ${PCL_LIBRARIES}
  ${PROJECT_NAME}_core_lib
)

# Semantic Test Harness Node
# Builds with TensorRT on GPU systems, or ONNXRuntime on CPU-only systems
if(USE_TENSORRT_YOLO AND ONNXRUNTIME_LIBRARY)
    message(STATUS "Building semantic_test_node (TensorRT + ONNXRuntime)")
    
    add_executable(semantic_test_node
        apps/semantic_test_node.cpp
        src/s_graphs/common/yolo_object_detector.cpp
        src/s_graphs/common/clip_feature_extractor.cpp
    )
    
    target_compile_definitions(semantic_test_node PRIVATE USE_TENSORRT_YOLO)
    target_include_directories(semantic_test_node PRIVATE
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )
    
    ament_target_dependencies(semantic_test_node
        rclcpp
        sensor_msgs
        visualization_msgs
        cv_bridge
        image_transport
    )
    
    target_link_libraries(semantic_test_node
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNXPARSER}
        ${CUDA_LIBRARIES}
        ${ONNXRUNTIME_LIBRARY}
        ${OpenCV_LIBRARIES}
    )
    
    install(TARGETS semantic_test_node
        DESTINATION lib/${PROJECT_NAME}
    )
elseif(ONNXRUNTIME_LIBRARY)
    message(STATUS "Building semantic_test_node (ONNXRuntime CPU-only)")
    
    add_executable(semantic_test_node
        apps/semantic_test_node.cpp
        src/s_graphs/common/yolo_onnx_detector.cpp
        src/s_graphs/common/clip_feature_extractor.cpp
    )
    
    target_include_directories(semantic_test_node PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )
    
    ament_target_dependencies(semantic_test_node
        rclcpp
        sensor_msgs
        visualization_msgs
        cv_bridge
        image_transport
    )
    
    target_link_libraries(semantic_test_node
        ${ONNXRUNTIME_LIBRARY}
        ${OpenCV_LIBRARIES}
    )
    
    install(TARGETS semantic_test_node
        DESTINATION lib/${PROJECT_NAME}
    )
else()
    message(WARNING "semantic_test_node will NOT be built — no inference backend found")
endif()

#############
## Install ##
#############

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}_core_lib g2o_${PROJECT_NAME}
  EXPORT export_s_graph_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS
  s_graphs_prefiltering_node s_graphs_scan_matching_odometry_node s_graphs_room_segmentation_node s_graphs_floor_plan_node s_graphs_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    ${PROJECT_NAME}/map2odom_publisher.py
    ${PROJECT_NAME}/tf_filter.py
    ${PROJECT_NAME}/iqa_ros.py
    ${PROJECT_NAME}/iqm.py
    scripts/zones_node.py
    scripts/metrics_recorder.py
    scripts/analyse_metrics.py
  DESTINATION lib/${PROJECT_NAME}
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  # the following lines skip linters
  set(ament_cmake_cppcheck_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(testPlane test/testPlane.cpp)
  target_link_libraries(testPlane ${PROJECT_NAME}_core_lib ${PCL_LIBRARIES} ${G2O_LIBRARIES})

  ament_add_gtest(testRoom test/testRoom.cpp)
  target_link_libraries(testRoom ${PROJECT_NAME}_core_lib ${PCL_LIBRARIES} ${G2O_LIBRARIES})

  # ament_add_gtest(testRoomCentreCompute test/testRoomCentreCompute.cpp src/s_graphs/room_utils.cpp)
  ament_add_gtest(testRoomCentreCompute test/testRoomCentreCompute.cpp)
  target_link_libraries(testRoomCentreCompute ${PROJECT_NAME}_core_lib ${PCL_LIBRARIES} ${G2O_LIBRARIES})

  install(TARGETS
    testPlane testRoom testRoomCentreCompute DESTINATION test/${PROJECT_NAME})
endif()

ament_export_dependencies(${DEPENDENCIES})
ament_export_include_directories(include)
ament_package()