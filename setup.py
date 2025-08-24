from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Manually specify PyTorch include and lib paths for local install
TORCH_BASE = os.path.dirname(torch.__file__)
TORCH_INCLUDE = os.path.join(TORCH_BASE, 'include')
TORCH_API_INCLUDE = os.path.join(TORCH_INCLUDE, 'torch', 'csrc', 'api', 'include')
TORCH_LIB = os.path.join(TORCH_BASE, 'lib')

ext_modules = [
    CppExtension(
        name="gpu_simulator_py",
        sources=[
            "src/pybind_wrapper.cpp",
            "src/gfloat.cpp", 
            "src/Hopper_simulator.cpp",
            "src/Ampere_simulator.cpp",
            "utils/utils.cpp"
        ],
        include_dirs=[
            "include",  # This will allow #include "Hopper_simulator.h" to work
            "utils",
            "src",
            TORCH_INCLUDE,
            TORCH_API_INCLUDE
        ],
        extra_compile_args=["-std=c++17", "-O3", "-DNDEBUG"],
        extra_link_args=["-Wl,-rpath," + TORCH_LIB],
    ),
]

setup(
    name="gpu_simulator_py",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.7",
) 