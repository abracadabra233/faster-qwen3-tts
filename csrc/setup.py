"""Build script for int8_gemv CUDA extension (Jetson AGX Orin, SM 8.7).

Includes:
  - int8_gemv.cu         : v1 dp4a GEMV kernel
  - fused_int8_gemv.cu   : v2 vectorized + fused W8A8 GEMV kernels
  - bindings.cpp         : PyTorch C++ bindings for all three entry points
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="int8_gemv_cuda",
    version="0.2.0",
    ext_modules=[
        CUDAExtension(
            name="int8_gemv_cuda",
            sources=["bindings.cpp", "int8_gemv.cu", "fused_int8_gemv.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_87,code=sm_87",
                    "--use_fast_math",
                    "-std=c++17",
                    "--threads=4",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
