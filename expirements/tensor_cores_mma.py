import cupy
import cupy.cuda
import torch
from typing import Optional
from torch import Tensor


bf16_wmma_ker = cupy.RawKernel(r'''
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
  #include <mma.h>
  using namespace nvcuda;
  __device__ void wmma_ker_dev(__nv_bfloat16 *a, __nv_bfloat16 *b, float *c, float *d) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    // wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(c_frag, c, 16, wmma::mem_row_major);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(d, c_frag, 16, wmma::mem_row_major);
  }
  extern "C" {
    __global__ void wmma_ker(__nv_bfloat16 *a, __nv_bfloat16 *b, float *c, float *d) {
      wmma_ker_dev(a,b,c,d);
    }
  }
 ''', 'wmma_ker', options=("-restrict","-lineinfo","-D__CUDA_NO_HALF_CONVERSIONS__"))


fp16_wmma_ker = cupy.RawKernel(r'''
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
  #include <mma.h>
  using namespace nvcuda;
  __device__ void wmma_ker_dev(half *a, half *b, float *c, float *d) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    // wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(c_frag, c, 16, wmma::mem_row_major);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(d, c_frag, 16, wmma::mem_row_major);
  }
  extern "C" {
    __global__ void wmma_ker(half *a, half *b, float *c, float *d) {
      wmma_ker_dev(a,b,c,d);
    }
  }
 ''', 'wmma_ker', options=("-restrict","-lineinfo","-D__CUDA_NO_HALF_CONVERSIONS__"))


def mma(left: Tensor, right: Tensor, acc: Optional[Tensor] = None, result_dtype: Optional[torch.dtype] = None) -> Tensor:
    assert left.device.type == 'cuda'
    assert left.numel() > 0
    assert left.dim() == right.dim()
    assert left.size(-1) == right.size(-2)
    assert left.dim() == 3

    # dst = torch.zeros(*left.shape[:-1], right.size(-1), dtype=left.dtype, device=left.device)
    # plan.run(left, right, dst, dst, print_module=False, stream=torch.cuda.current_stream().cuda_stream)

    # dst = torch.matmul(left, right)
    if acc is None:
        acc = torch.zeros(*left.shape[:-1], right.size(-1), dtype=torch.float32, device=left.device)
    else:
        acc = acc.to(torch.float32)

    dst = torch.zeros(*left.shape[:-1], right.size(-1), dtype=torch.float32, device=left.device)

    with cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        for batch in range(dst.size(0)):
          if left.dtype == torch.float8_e4m3fn:
            left_ = cupy.asarray(left[batch].clone().view(torch.uint8))
            right_ = cupy.asarray(right[batch].clone().view(torch.uint8).contiguous().mT)
          else:
            left_ = cupy.asarray(left[batch].clone())
            # dlpack = torch.utils.dlpack.to_dlpack(left[batch].clone())
            # left_ = cupy.fromDlpack(dlpack)
            right_ = cupy.asarray(right[batch].clone())
            # dlpack = torch.utils.dlpack.to_dlpack(right[batch].clone())
            # right_ = cupy.fromDlpack(dlpack)
          acc_ = cupy.asarray(acc[batch])
          dst_ = cupy.asarray(dst[batch])
          if left.dtype == torch.bfloat16:
              bf16_wmma_ker((1,1), (32,1), (left_, right_, acc_, dst_))  # grid, block and arguments
          elif left.dtype == torch.float16:
              fp16_wmma_ker((1,1), (32,1), (left_, right_, acc_, dst_))  # grid, block and arguments
          else:
              raise NotImplementedError(f'{left.dtype=}')
    if result_dtype is not None:
        dst = dst.to(result_dtype)

    return dst

if __name__ == "__main__":
    left = torch.zeros(1,8, 32, dtype=torch.bfloat16, device="cuda")
    right = torch.zeros(1,8, 32, dtype=torch.bfloat16, device="cuda").mT
    left[0, 0, 0] = 1
    right[0, 0, 0] = 1
    left = left.to(torch.float8_e4m3fn)
    right = right.to(torch.float8_e4m3fn)
    result = mma(left, right)
    print(result[0, 0, 0].item())