cmake_minimum_required(VERSION 3.8)
project(tracking_of)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(situational_graphs_msgs REQUIRED)

# Try to find OpenCV - handle both OpenCV3 and OpenCV4
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3 REQUIRED)
endif()

# Set dependencies list for ament_target_dependencies
set(DEPENDENCIES
  rclcpp
  sensor_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  cv_bridge
  situational_graphs_msgs)

# add executable
add_executable(of_track_node src/of_track.cpp)

target_include_directories(of_track_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

ament_target_dependencies(of_track_node ${DEPENDENCIES})

target_link_libraries(of_track_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES})

# install targets
install(TARGETS of_track_node
  DESTINATION lib/${PROJECT_NAME})

# install headers
install(DIRECTORY include/
  DESTINATION include)

# install launch files if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})
endif()

# install config files if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME})
endif()

# install rviz files if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rviz")
  install(DIRECTORY rviz
    DESTINATION share/${PROJECT_NAME})
endif()

if(BUILD_TESTING)
  # Optional: Add unit tests here if needed
  # find_package(ament_cmake_gtest REQUIRED)
endif()

ament_package()
