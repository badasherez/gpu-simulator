#include "../utils/utils.h"
#include <cstring>

GfloatTensor bfloat16_to_gfloat_tensor(const torch::Tensor& tensor) {
    // Check if tensor is bfloat16
    if (tensor.dtype() != torch::kBFloat16) {
        throw std::invalid_argument("Input tensor must be bfloat16 dtype");
    }
    
    // Get tensor size and shape
    auto num_elements = tensor.numel();
    auto tensor_shape = tensor.sizes().vec();
    
    std::vector<Gfloat> result;
    result.reserve(num_elements);
    
    // Access tensor data
    auto tensor_data = tensor.data_ptr<c10::BFloat16>();
    
    for (int64_t i = 0; i < num_elements; ++i) {
        c10::BFloat16 bf16 = tensor_data[i];
        
        // Extract components from bfloat16
        uint16_t bits = bf16.x;
        
        // Extract sign (bit 15)
        bool sign = (bits >> 15) & 1;
        
        // Extract exponent bits (bits 14-7)
        uint8_t exp_bits = (bits >> 7) & 0xFF;
        
        // Extract significand bits (bits 6-0)
        uint8_t sig_bits = bits & 0x7F;
        int16_t actual_exponent = static_cast<int16_t>(exp_bits) - 127;
            
        // Add implicit leading 1 to significand
        int32_t significand = (exp_bits != 0) ? (sig_bits | 0x80) : sig_bits;
            
        result.emplace_back(sign, actual_exponent, significand);
    }
          
    return GfloatTensor(result, tensor_shape);
}

Gfloat multiply_bfloat16_to_gfloat(uint16_t a, uint16_t b, int16_t zero_exponent) {
    // Extract components from first bfloat16 (a)
    bool sign_a = (a >> 15) & 1;
    uint8_t exp_bits_a = (a >> 7) & 0xFF;
    uint8_t sig_bits_a = a & 0x7F;
    
    // Extract components from second bfloat16 (b)
    bool sign_b = (b >> 15) & 1;
    uint8_t exp_bits_b = (b >> 7) & 0xFF;
    uint8_t sig_bits_b = b & 0x7F;
    

    // Calculate significands with implicit leading 1
    int32_t significand_a = (exp_bits_a != 0) ? (sig_bits_a | 0x80) : sig_bits_a;
    int32_t significand_b = (exp_bits_b != 0) ? (sig_bits_b | 0x80) : sig_bits_b;
    if (exp_bits_a == 0) {
        exp_bits_a = 1;
    }
    if (exp_bits_b == 0) {
        exp_bits_b = 1;
    }   
    // Multiply significands
    int32_t result_significand = (significand_a * significand_b)<<9;
    

    int16_t result_exponent = exp_bits_a + exp_bits_b - 254;
    
    // XOR the signs
    bool result_sign = sign_a ^ sign_b;
    
    // Handle zero result
    if (result_significand == 0) {
        return Gfloat(result_sign, zero_exponent, 0);
    }
    
    return Gfloat(result_sign, result_exponent, result_significand);
}

Gfloat multiply_float16_to_gfloat(uint16_t a, uint16_t b, int16_t zero_exponent) {
    // Extract components from first float16 (a)
    // Float16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
    bool sign_a = (a >> 15) & 1;
    uint8_t exp_bits_a = (a >> 10) & 0x1F;  // 5 bits for exponent
    uint16_t sig_bits_a = a & 0x3FF;        // 10 bits for mantissa
    
    // Extract components from second float16 (b)
    bool sign_b = (b >> 15) & 1;
    uint8_t exp_bits_b = (b >> 10) & 0x1F;  // 5 bits for exponent
    uint16_t sig_bits_b = b & 0x3FF;        // 10 bits for mantissa
    

    // Calculate significands with implicit leading 1
    // For float16, add implicit leading 1 if not denormalized (exp != 0)
    int32_t significand_a = (exp_bits_a != 0) ? (sig_bits_a | 0x400) : sig_bits_a;  // 0x400 = 2^10
    int32_t significand_b = (exp_bits_b != 0) ? (sig_bits_b | 0x400) : sig_bits_b;  // 0x400 = 2^10
    if (exp_bits_a == 0) {
        exp_bits_a = 1;
    }
    if (exp_bits_b == 0) {
        exp_bits_b = 1;
    }   
    
    // Multiply significands with shift by 3 (instead of 9 for bfloat16)
    int32_t result_significand = (significand_a * significand_b) << 3;
    

    // Calculate result exponent with float16 bias (15)
    // Float16 exponent bias is 15, so we subtract 2*15 = 30
    int16_t result_exponent = exp_bits_a + exp_bits_b - 30;
    
    // XOR the signs
    bool result_sign = sign_a ^ sign_b;
    
    // Handle zero result
    if (result_significand == 0) {
        return Gfloat(result_sign, zero_exponent, 0);
    }

    return Gfloat(result_sign, result_exponent, result_significand);
}

