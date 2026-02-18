import os
import sys
import cupy
import cupy.cuda
import torch
from typing import Optional
from torch import Tensor

# Try to import the FP8 E4M3 WGMMA extension
_fp8_e4m3_ext = None
def _get_fp8_e4m3_ext():
    global _fp8_e4m3_ext
    if _fp8_e4m3_ext is None:
        ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wgmma_e4m3')
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        try:
            import fp8_e4m3_wgmma_ext
            _fp8_e4m3_ext = fp8_e4m3_wgmma_ext
        except ImportError:
            raise ImportError(
                "fp8_e4m3_wgmma_ext not built. "
                "Run: cd experiments/wgmma_e4m3 && make build"
            )
    return _fp8_e4m3_ext


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


def mma_fp8_e4m3(left: Tensor, right: Tensor, acc: Optional[Tensor] = None) -> Tensor:
    """
    FP8 E4M3 matrix multiply-accumulate using Hopper WGMMA.

    Uses wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
    The WGMMA kernel computes: D = A @ B^T + C

    Convention (different from bf16/fp16 due to WGMMA layout):
        left:  [batch, M, K] float8_e4m3fn — A matrix, row-major
        right: [batch, N, K] float8_e4m3fn — B matrix (N x K), contiguous
        acc:   Optional [batch, M, N] float32 accumulator

    Computes: D[m, n] = sum_k left[b, m, k] * right[b, n, k] + acc[b, m, n]
              (i.e., D = A @ B^T + C)

    WGMMA constraints: M <= 64, N <= 128, K <= 32
    Inputs are automatically zero-padded to [64, 32] and [128, 32].

    Example:
        left = torch.zeros(1, 64, 32, dtype=torch.float8_e4m3fn, device="cuda")
        right = torch.zeros(1, 128, 32, dtype=torch.float8_e4m3fn, device="cuda")
        left[0, 0, 0] = 1.0;  left[0, 0, 1] = 1.0
        right[0, 0, 0] = 1.0; right[0, 0, 1] = -1.0
        # D[0,0] = 1*1 + 1*(-1) = 0
        result = mma_fp8_e4m3(left, right)

    Returns:
        [batch, M, N] float32 result
    """
    ext = _get_fp8_e4m3_ext()

    assert left.device.type == 'cuda'
    assert left.dtype == torch.float8_e4m3fn
    assert left.dim() == 3 and right.dim() == 3

    batch_size = left.size(0)
    M, K = left.size(1), left.size(2)
    N = right.size(-2)
    K_r = right.size(-1)
    assert K == K_r, f"K mismatch: left K={K}, right K={K_r}"
    assert M <= 64, f"M={M} exceeds WGMMA max 64"
    assert N <= 128, f"N={N} exceeds WGMMA max 128"
    assert K <= 32, f"K={K} exceeds WGMMA max 32"

    dst = torch.zeros(batch_size, M, N, dtype=torch.float32, device=left.device)

    for b in range(batch_size):
        # Pad A to [64, 32]
        a_pad = torch.zeros(64, 32, dtype=torch.float8_e4m3fn, device=left.device)
        a_pad[:M, :K] = left[b].contiguous()

        # Pad B to [128, 32] — right[b] is [N, K], make contiguous
        b_pad = torch.zeros(128, 32, dtype=torch.float8_e4m3fn, device=left.device)
        b_pad[:N, :K] = right[b].contiguous()

        # Pad accumulator to [64, 128]
        c_pad = None
        if acc is not None:
            c_pad = torch.zeros(64, 128, dtype=torch.float32, device=left.device)
            c_pad[:M, :N] = acc[b].float()

        # Call WGMMA kernel
        d_full = ext.fp8_e4m3_wgmma(a_pad, b_pad, c_pad)

        # Extract the relevant output region
        dst[b] = d_full[:M, :N]

    return dst


def mma(left: Tensor, right: Tensor, acc: Optional[Tensor] = None, result_dtype: Optional[torch.dtype] = None) -> Tensor:
    assert left.device.type == 'cuda'
    assert left.numel() > 0
    assert left.dim() == right.dim()
    assert left.dim() == 3

    assert left.size(-1) == right.size(-2)

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
            left_ = cupy.asarray(left[batch].clone())
            right_ = cupy.asarray(right[batch].clone())
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