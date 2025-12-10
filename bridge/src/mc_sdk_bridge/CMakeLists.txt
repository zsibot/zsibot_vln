
cmake_minimum_required(VERSION 3.10)
project(mc_sdk_bridge)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mc_sdk_msg REQUIRED)
find_package(zsibot_common REQUIRED)

# 找 libzmq
find_library(ZMQ_LIB zmq REQUIRED)

file(GLOB_RECURSE sources "src/*.cpp")
add_executable(mc_sdk_bridge ${sources})

target_include_directories(mc_sdk_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(mc_sdk_bridge PRIVATE /usr/local/include)

ament_target_dependencies(mc_sdk_bridge rclcpp std_msgs geometry_msgs mc_sdk_msg)

target_link_libraries(mc_sdk_bridge
  zsibot_common::zsibot_base
  zsibot_common::zsibot_rmw_ecal
  zsibot_common::robot_sdk
  ${ZMQ_LIB}      # 链接 libzmq
)

install(TARGETS mc_sdk_bridge
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include/
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()


