#ifndef CUSTOM_SIMULATOR_H
#define CUSTOM_SIMULATOR_H

#include <vector>
#include <torch/torch.h>
#include "../include/gfloat.h"
#include "../utils/utils.h"

/**
 * @brief Custom GPU Simulator class
 * 
 * This class simulates a custom GPU architecture with user-configured parameters.
 * Configure by modifying CUSTOM_* constants in utils/utils.h based on your
 * GPU characterization results from GPU_reproduction.py
 */
class Custom_simulator {
public:
    // Constructor
    Custom_simulator();

    // Destructor
    ~Custom_simulator() = default;

    // Group sum function
    Gfloat group_sum(const std::vector<Gfloat>& array);
    
    // Matrix multiplication using bfloat16 to Gfloat conversion
    torch::Tensor matmul_bfloat16(const torch::Tensor& A, const torch::Tensor& B);
    
    // Matrix multiplication using float16 to Gfloat conversion
    torch::Tensor matmul_float16(const torch::Tensor& A, const torch::Tensor& B);

    // Unified matrix multiplication that dispatches based on tensor dtype
    torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B);

private:
    // Template helper function for matrix multiplication
    template<typename T>
    torch::Tensor matrix_multiply_template(const torch::Tensor& A, const torch::Tensor& B, 
                                         torch::ScalarType expected_dtype, 
                                         const std::string& dtype_name,
                                         Gfloat (*multiply_func)(uint16_t, uint16_t, int16_t),
                                         int16_t zero_exponent);
};

#endif // CUSTOM_SIMULATOR_H

