cmake_minimum_required(VERSION 2.8.9)
project(lenet)

add_definitions(-std=c++11)

# set(TARGET_NAME "lenet")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(include)
include_directories(${PROJECT_SOURCE_DIR}/include)
# Tìm gói OpenCV 1
find_package(OpenCV REQUIRED)
# Bao gồm thư viện OpenCV 1
include_directories(${OpenCV_INCLUDE_DIRS})
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# file(GLOB SOURCES "src/*.cpp")
FILE(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/include/*.hpp ${PROJECT_SOURCE_DIR}/include/*.h)

add_executable(main ${SRC_FILES})
# Liên kết với thư viện OpenCV 1
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main nvinfer)
target_link_libraries(main cudart)