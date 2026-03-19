#!/usr/bin/env python3
import shutil
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

host_compiler = (
    shutil.which('g++-10')
    or shutil.which('g++-9')
    or shutil.which('g++-8')
    or shutil.which('g++')
)

print(f'Using host compiler: {host_compiler}')

cxx_args = ['-std=c++14']
nvcc_args = [
    '-std=c++14',
    '-gencode', 'arch=compute_86,code=sm_86',
]

if host_compiler:
    nvcc_args = ['-ccbin', host_compiler] + nvcc_args

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension(
            'correlation_cuda',
            ['correlation_cuda.cc', 'correlation_cuda_kernel.cu'],
            include_dirs=['/usr/include'],
            library_dirs=['/usr/lib/x86_64-linux-gnu'],
            extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args},
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
