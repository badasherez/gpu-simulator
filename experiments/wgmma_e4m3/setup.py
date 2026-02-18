from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fp8_e4m3_wgmma',
    ext_modules=[
        CUDAExtension(
            name='fp8_e4m3_wgmma_ext',
            sources=[
                'fp8_e4m3_wgmma_ext.cpp',
                'fp8_e4m3_wgmma.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode', 'arch=compute_90a,code=sm_90a',
                    '--use_fast_math',
                    '-std=c++17',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

