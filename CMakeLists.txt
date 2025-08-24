cmake_minimum_required(VERSION 3.10)
project(gpu_simulator)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(include)

# Main executable
add_executable(gpu_simulator src/main.cpp src/gfloat.cpp utils/utils.cpp)

# Python module
pybind11_add_module(gpu_simulator_py 
    src/pybind_wrapper.cpp 
    src/gfloat.cpp 
    src/Hopper_simulator.cpp 
    utils/utils.cpp)

# Link libraries
target_link_libraries(gpu_simulator_py PRIVATE ${TORCH_LIBRARIES})

# Set include directories for the Python module
target_include_directories(gpu_simulator_py PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    include
    utils)
