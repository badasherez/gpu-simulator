#ifndef HOPPER_FP8_SIMULATOR_H
#define HOPPER_FP8_SIMULATOR_H

#include <vector>
#include <torch/torch.h>
#include "../include/gfloat.h"
#include "../utils/utils.h"

/**
 * @brief Hopper FP8 E4M3 Simulator (QGMMA instruction)
 * 
 * Simulates the QGMMA.64xNx32.F32.E4M3.E4M3 tensor core instruction.
 * 
 * Key differences from the bf16/fp16 Hopper simulator (HMMA):
 *   - Input format: float8_e4m3fn (4 exp bits, 3 mantissa bits, bias=7)
 *   - Internal significand: 14 bits (vs 26 for bf16)
 *   - Single accumulation group of 33 (acc + 32 products, no sub-grouping)
 *   - Subnormals NOT normalized before multiplication
 *   - Output rounding: towards_0
 *   - Zero exponent: -139 (vs -133 for bf16)
 */
class Hopper_fp8_simulator {
public:
    Hopper_fp8_simulator();
    ~Hopper_fp8_simulator() = default;

    // Group sum: accumulates all elements in a single group (no sub-grouping)
    Gfloat group_sum(const std::vector<Gfloat>& array);
    
    // Matrix multiplication: A [M,K] @ B^T [N,K] → D [M,N] in float32
    // Both A and B are float8_e4m3fn contiguous tensors
    torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B);
};

#endif // HOPPER_FP8_SIMULATOR_H

