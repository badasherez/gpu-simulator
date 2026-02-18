#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <torch/torch.h>
#include "../include/gfloat.h"

// Constants for Hopper architecture
constexpr int16_t HOPPER_ZERO_EXPONENT = -133;
constexpr int HOPPER_SIGNIFICAND_WIDTH = 26;
constexpr int HOPPER_ACCUMULATOR_GROUP_SIZE = 17;

// Constants for Ampere architecture
constexpr int16_t AMPERE_ZERO_EXPONENT = -132;  // Different from Hopper
constexpr int AMPERE_SIGNIFICAND_WIDTH = 25;
constexpr int AMPERE_ACCUMULATOR_GROUP_SIZE = 9;

// Constants for Hopper FP8 E4M3 (QGMMA) architecture
// Characterized via GPU_reproduction_fp8.py on H100
constexpr int16_t HOPPER_FP8_ZERO_EXPONENT = -139;
constexpr int HOPPER_FP8_SIGNIFICAND_WIDTH = 14;          // 13 fractional + 1 implicit
constexpr int HOPPER_FP8_ACCUMULATOR_GROUP_SIZE = 33;     // acc + 32 products, single group

// Constants for FP32 format
constexpr int FP32_SIGNIFICAND_WIDTH = 24;
constexpr int16_t FP32_MIN_NONZERO_EXPONENT = -126;

// Constants for Custom GPU (user-configurable)
// Update these values based on your GPU_reproduction.py output:
// - CUSTOM_ZERO_EXPONENT: Use "Minimal max exponent" from output
// - CUSTOM_SIGNIFICAND_WIDTH: Use "Internal precision" + 1
// - CUSTOM_ACCUMULATOR_GROUP_SIZE: Use "Accumulation group size"
constexpr int16_t CUSTOM_ZERO_EXPONENT = -133;          // Default: H100 value
constexpr int CUSTOM_SIGNIFICAND_WIDTH = 26;            // Default: H100 value (25 + 1)
constexpr int CUSTOM_ACCUMULATOR_GROUP_SIZE = 17;       // Default: H100 value

struct GfloatTensor {
    std::vector<Gfloat> data;
    std::vector<int64_t> shape;
    
    GfloatTensor(const std::vector<Gfloat>& d, const std::vector<int64_t>& s) 
        : data(d), shape(s) {}
};

/**
 * @brief Converts a PyTorch tensor of bfloat16 to GfloatTensor with same shape
 * 
 * @param tensor Input PyTorch tensor with bfloat16 dtype
 * @return GfloatTensor Contains data and shape information
 */
GfloatTensor bfloat16_to_gfloat_tensor(const torch::Tensor& tensor);

/**
 * @brief Converts a PyTorch tensor of bfloat16 to a vector of Gfloats
 * 
 * @param tensor Input PyTorch tensor with bfloat16 dtype
 * @return std::vector<Gfloat> Vector of converted Gfloat values
 */
std::vector<Gfloat> bfloat16_to_gfloat(const torch::Tensor& tensor);

/**
 * @brief Converts a vector of Gfloats to a PyTorch tensor of bfloat16
 * 
 * @param gfloats Input vector of Gfloat values
 * @return torch::Tensor PyTorch tensor with bfloat16 dtype
 */
torch::Tensor gfloat_to_bfloat16(const std::vector<Gfloat>& gfloats);

/**
 * @brief Multiplies two bfloat16 values and returns a Gfloat result
 * 
 * @param a First bfloat16 value
 * @param b Second bfloat16 value
 * @param zero_exponent The zero exponent value for the target architecture
 * @return Gfloat Result of the multiplication
 */
Gfloat multiply_bfloat16_to_gfloat(uint16_t a, uint16_t b, int16_t zero_exponent);

/**
 * @brief Multiplies two float16 values and returns a Gfloat result
 * 
 * @param a First float16 value (5 exp bits, 10 mantissa bits)
 * @param b Second float16 value (5 exp bits, 10 mantissa bits)
 * @param zero_exponent The zero exponent value for the target architecture
 * @return Gfloat Result of the multiplication
 */
Gfloat multiply_float16_to_gfloat(uint16_t a, uint16_t b, int16_t zero_exponent);

/**
 * @brief Multiplies two float8_e4m3 values and returns a Gfloat result
 * 
 * E4M3fn format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
 * Subnormals are NOT normalized (kept denormalized).
 * Product of two 4-bit significands = 8-bit result, shifted to fill Gfloat.
 * 
 * @param a First float8_e4m3 value (raw byte)
 * @param b Second float8_e4m3 value (raw byte)
 * @param zero_exponent The zero exponent value for the target architecture
 * @return Gfloat Result of the multiplication
 */
Gfloat multiply_fp8e4m3_to_gfloat(uint8_t a, uint8_t b, int16_t zero_exponent);

#endif // UTILS_H
