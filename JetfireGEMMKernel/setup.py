import os
from setuptools import setup, find_packages
import torch
from torch.utils import cpp_extension

compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 10 + compute_capability[1]

setup(
    name='BlockQuantize',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='BlockQuantizeCUDA',
            sources=[
                'BlockQuantize/cpp_extension/igemm/igemm_BlockSquare32OutputIntQuantizeStochastic.cu',
                'BlockQuantize/cpp_extension/igemm/igemm_BlockSquare32OutputIntQuantizeBiasRowCol.cu',
                'BlockQuantize/cpp_extension/igemm/igemm_BlockSquare32OutputIntQuantizeBiasRowRow.cu',
                'BlockQuantize/cpp_extension/igemm/igemm_BlockSquare32OutputIntQuantize.cu',
                'BlockQuantize/cpp_extension/igemm/igemm_BlockSquare32OutputIntQuantizeTrash.cu',
                'BlockQuantize/cpp_extension/igemm/igemm_BlockSquare32OutputFp.cu',
                'BlockQuantize/cpp_extension/igemm/igemm_BasicInt8Gemm.cu',
                'BlockQuantize/cpp_extension/hgemm.cu',
                'BlockQuantize/cpp_extension/bindings.cpp'
            ],
            include_dirs=['BlockQuantize/cpp_extension/include',],
            extra_compile_args={
                "nvcc": [f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}", 
                          "-use_fast_math", 
                        #   "-O3",
                          '-U__CUDA_NO_HALF_OPERATORS__', 
                          '-U__CUDA_NO_HALF_CONVERSIONS__', 
                          '-U__CUDA_NO_HALF2_OPERATORS__']
            },
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']
    ),
)
