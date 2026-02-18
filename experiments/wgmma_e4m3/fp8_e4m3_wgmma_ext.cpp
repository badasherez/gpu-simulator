#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

void launch_fp8_e4m3_wgmma(
    const void* a,
    const void* b,
    const void* c,
    void* d,
    int m, int n, int k,
    cudaStream_t stream);

torch::Tensor fp8_e4m3_wgmma(
    torch::Tensor a,  // [64, 32] float8_e4m3fn
    torch::Tensor b,  // [128, 32] float8_e4m3fn
    torch::optional<torch::Tensor> c)  // [64, 128] float32 (optional)
{
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");
  TORCH_CHECK(a.dim() == 2 && a.size(0) == 64 && a.size(1) == 32,
              "a must be [64, 32], got [" + std::to_string(a.size(0)) + ", " + std::to_string(a.size(1)) + "]");
  TORCH_CHECK(b.dim() == 2 && b.size(0) == 128 && b.size(1) == 32,
              "b must be [128, 32], got [" + std::to_string(b.size(0)) + ", " + std::to_string(b.size(1)) + "]");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "a must be float8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "b must be float8_e4m3fn");

  const void* c_ptr = nullptr;
  if (c.has_value()) {
    auto c_tensor = c.value();
    TORCH_CHECK(c_tensor.is_cuda() && c_tensor.is_contiguous(),
                "c must be contiguous CUDA tensor");
    TORCH_CHECK(c_tensor.dim() == 2 && c_tensor.size(0) == 64 && c_tensor.size(1) == 128,
                "c must be [64, 128]");
    TORCH_CHECK(c_tensor.scalar_type() == torch::kFloat32, "c must be float32");
    c_ptr = c_tensor.data_ptr();
  }

  auto d = torch::zeros({64, 128},
      torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));

  launch_fp8_e4m3_wgmma(
    a.data_ptr(),
    b.data_ptr(),
    c_ptr,
    d.data_ptr(),
    64, 128, 32,
    at::cuda::getCurrentCUDAStream()
  );

  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_e4m3_wgmma", &fp8_e4m3_wgmma,
        "FP8 E4M3 WGMMA m64n128k32.f32.e4m3.e4m3\n"
        "Computes D = A @ B^T + C\n"
        "  A: [64, 32] float8_e4m3fn\n"
        "  B: [128, 32] float8_e4m3fn\n"
        "  C: [64, 128] float32 (optional)\n"
        "  D: [64, 128] float32\n",
        py::arg("a"), py::arg("b"), py::arg("c") = py::none());
}

