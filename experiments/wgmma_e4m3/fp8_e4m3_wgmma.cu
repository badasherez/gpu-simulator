#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// FP8 E4M3 WGMMA kernel: m64n128k32.f32.e4m3.e4m3
// Computes D = A @ B^T + C where:
//   A is [64, 32] e4m3 (row-major)
//   B is [128, 32] e4m3 (row-major, i.e. N x K)
//   C is [64, 128] f32 (optional accumulator)
//   D is [64, 128] f32

typedef __nv_fp8_e4m3 e4m3;

union GmmaDescriptor {
  __device__ constexpr GmmaDescriptor() noexcept : desc_(0) {}
  __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept : desc_(desc) {}

  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];

  struct {
    uint16_t start_address_ : 14, : 2;
    uint16_t leading_byte_offset_ : 14, : 2;
    uint16_t stride_byte_offset_ : 14, : 2;
    uint8_t : 1, base_offset_ : 3, : 4;
    uint8_t : 6, layout_type_ : 2;
  } bitfield;

  __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

template <class PointerType>
__device__ GmmaDescriptor make_desc(PointerType smem_ptr) {
  GmmaDescriptor desc;
  uint32_t uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc.bitfield.start_address_ = uint_ptr >> 4;
  desc.bitfield.layout_type_ = 0;
  desc.bitfield.leading_byte_offset_ = 8;
  desc.bitfield.stride_byte_offset_ = 16;
  desc.bitfield.base_offset_ = 0;
  return desc;
}

// WGMMA m64n128k32.f32.e4m3.e4m3 — 64 f32 output registers per thread
__device__ void MMA(uint64_t const &desc_a, uint64_t const &desc_b,
    float &d00, float &d01, float &d02, float &d03,
    float &d04, float &d05, float &d06, float &d07,
    float &d08, float &d09, float &d10, float &d11,
    float &d12, float &d13, float &d14, float &d15,
    float &d16, float &d17, float &d18, float &d19,
    float &d20, float &d21, float &d22, float &d23,
    float &d24, float &d25, float &d26, float &d27,
    float &d28, float &d29, float &d30, float &d31,
    float &d32, float &d33, float &d34, float &d35,
    float &d36, float &d37, float &d38, float &d39,
    float &d40, float &d41, float &d42, float &d43,
    float &d44, float &d45, float &d46, float &d47,
    float &d48, float &d49, float &d50, float &d51,
    float &d52, float &d53, float &d54, float &d55,
    float &d56, float &d57, float &d58, float &d59,
    float &d60, float &d61, float &d62, float &d63)
{
    int scale_D = 1;
    constexpr int32_t scaleA = 1;
    constexpr int32_t scaleB = 1;
    asm volatile("{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %66, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
    "{%0,   %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
    " %8,   %9,  %10, %11, %12, %13, %14, %15, "
    " %16,  %17, %18, %19, %20, %21, %22, %23, "
    " %24,  %25, %26, %27, %28, %29, %30, %31, "
    " %32,  %33, %34, %35, %36, %37, %38, %39, "
    " %40,  %41, %42, %43, %44, %45, %46, %47, "
    " %48,  %49, %50, %51, %52, %53, %54, %55, "
    " %56,  %57, %58, %59, %60, %61, %62, %63},"
    " %64,"
    " %65,"
    " p,   %67, %68;\n"
    "}\n"
    : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03),
      "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
      "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
      "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
      "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19),
      "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
      "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27),
      "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31),
      "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35),
      "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39),
      "+f"(d40), "+f"(d41), "+f"(d42), "+f"(d43),
      "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
      "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51),
      "+f"(d52), "+f"(d53), "+f"(d54), "+f"(d55),
      "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59),
      "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
    : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_D)),
      "n"(int32_t(scaleA)), "n"(int32_t(scaleB)));
}

__global__ void fp8_e4m3_wgmma_kernel(
    const e4m3* __restrict__ a,      // [64, 32] E4M3
    const e4m3* __restrict__ b,      // [128, 32] E4M3
    const float* __restrict__ c,     // [64, 128] F32 (optional accumulator)
    float* __restrict__ d,           // [64, 128] F32 output
    int m, int n, int k)
{
  __shared__ e4m3 A[64 * 32];
  __shared__ e4m3 B[128 * 32];

  int tid = threadIdx.x;
  int wid = tid / 32;
  int lid = tid % 32;

  // Load A [64, 32] with 128B swizzle layout
  for (int i = 0; i < 16; ++i) {
    int r = tid / 2;
    int col = tid % 2 * 16 + i;
    A[r / 8 * 8 * 32 + col / 16 * 8 * 16 + r % 8 * 16 + col % 16] = a[r * 32 + col];
  }

  // Load B [128, 32] with 128B swizzle layout
  for (int i = 0; i < 32; ++i) {
    int r = tid;
    int col = i;
    B[r / 8 * 8 * 32 + col / 16 * 8 * 16 + r % 8 * 16 + col % 16] = b[r * 32 + col];
  }

  __syncthreads();

  GmmaDescriptor desc_a = make_desc(A);
  GmmaDescriptor desc_b = make_desc(B);

  // f32 accumulator: 64 registers per thread for m64n128
  float accum[64];
  for (int i = 0; i < 64; ++i) {
    accum[i] = 0.0f;
  }

  // Load accumulator from C if provided
  // Thread mapping for f32 m64n128:
  //   For f32, each pair of consecutive registers corresponds to the two f16
  //   values that would have been packed in one uint32_t in the f16 variant.
  //   register = 4*c + 2*r + h  (c ∈ 0..15, r ∈ 0..1, h ∈ 0..1)
  //   row = wid*16 + r*8 + lid/4
  //   col = c*8 + 2*(lid%4) + h
  if (c != nullptr) {
    for (int r = 0; r < 2; ++r) {
      for (int ci = 0; ci < 16; ++ci) {
        int row = wid * 16 + r * 8 + lid / 4;
        int col_base = ci * 8 + 2 * (lid % 4);
        accum[4 * ci + 2 * r]     = c[row * 128 + col_base];
        accum[4 * ci + 2 * r + 1] = c[row * 128 + col_base + 1];
      }
    }
  }

  // Compiler barriers to prevent reordering
  for (int i = 0; i < 64; ++i) {
    asm volatile("" : "+f"(accum[i]));
  }

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

  MMA(desc_a, desc_b,
      accum[0],  accum[1],  accum[2],  accum[3],
      accum[4],  accum[5],  accum[6],  accum[7],
      accum[8],  accum[9],  accum[10], accum[11],
      accum[12], accum[13], accum[14], accum[15],
      accum[16], accum[17], accum[18], accum[19],
      accum[20], accum[21], accum[22], accum[23],
      accum[24], accum[25], accum[26], accum[27],
      accum[28], accum[29], accum[30], accum[31],
      accum[32], accum[33], accum[34], accum[35],
      accum[36], accum[37], accum[38], accum[39],
      accum[40], accum[41], accum[42], accum[43],
      accum[44], accum[45], accum[46], accum[47],
      accum[48], accum[49], accum[50], accum[51],
      accum[52], accum[53], accum[54], accum[55],
      accum[56], accum[57], accum[58], accum[59],
      accum[60], accum[61], accum[62], accum[63]);

  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

  for (int i = 0; i < 64; ++i) {
    asm volatile("" : "+f"(accum[i]));
  }

  __syncthreads();

  // Store f32 output: 64x128 float
  // Same interleaved mapping as the accumulator load
  for (int r = 0; r < 2; ++r) {
    for (int ci = 0; ci < 16; ++ci) {
      int row = wid * 16 + r * 8 + lid / 4;
      int col_base = ci * 8 + 2 * (lid % 4);
      d[row * 128 + col_base]     = accum[4 * ci + 2 * r];
      d[row * 128 + col_base + 1] = accum[4 * ci + 2 * r + 1];
    }
  }
}

void launch_fp8_e4m3_wgmma(
    const void* a,
    const void* b,
    const void* c,
    void* d,
    int m, int n, int k,
    cudaStream_t stream)
{
  dim3 grid(1);
  dim3 block(128);

  fp8_e4m3_wgmma_kernel<<<grid, block, 0, stream>>>(
    reinterpret_cast<const e4m3*>(a),
    reinterpret_cast<const e4m3*>(b),
    reinterpret_cast<const float*>(c),
    reinterpret_cast<float*>(d),
    m, n, k
  );
}

