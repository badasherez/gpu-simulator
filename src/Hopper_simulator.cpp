#include "../include/Hopper_simulator.h"
#include <bit>  // C++20
#include <thread>
#include <mutex>
#include "../utils/utils.h"

// Constructor
Hopper_simulator::Hopper_simulator() {
    // Empty constructor
}

// Group sum function
Gfloat Hopper_simulator::group_sum(const std::vector<Gfloat>& array) {
    // Calculate shift value from Hopper to FP32 significand width
    constexpr int hopper_to_fp32_shift = HOPPER_SIGNIFICAND_WIDTH - FP32_SIGNIFICAND_WIDTH;
    
    // Step 1: Find the maximal exponent of the Gfloats
    int16_t max_exponent = HOPPER_ZERO_EXPONENT;
    
    for (const auto& gfloat : array) {
        if (gfloat.get_exponent() > max_exponent) {
            max_exponent = gfloat.get_exponent();
        }
    }
    
    int32_t significand = 0;
    for (const auto& gfloat : array) {
        int shift = (max_exponent - gfloat.get_exponent());
        if(shift >= 32) {
            continue;
        }
        if (gfloat.get_sign()) {
            significand -= (gfloat.get_significand() << hopper_to_fp32_shift) >> (max_exponent - gfloat.get_exponent());
        } else {
            significand += (gfloat.get_significand() << hopper_to_fp32_shift) >> (max_exponent - gfloat.get_exponent());
        }
    }
    bool sign = significand < 0;
    significand = std::abs(significand);
    int width = significand == 0 ? 0 : 32 - __builtin_clz(significand);
    if (width == 0) {
        return Gfloat(sign, HOPPER_ZERO_EXPONENT, 0);
    }
    int16_t exp = max_exponent + width - HOPPER_SIGNIFICAND_WIDTH;
    if (width > HOPPER_SIGNIFICAND_WIDTH) {
        significand = significand >> (width - HOPPER_SIGNIFICAND_WIDTH);
    }
    else {
        significand = significand << (HOPPER_SIGNIFICAND_WIDTH - width);
    }
    if (exp < FP32_MIN_NONZERO_EXPONENT) {
        significand = significand >> (FP32_MIN_NONZERO_EXPONENT - exp);
        exp = FP32_MIN_NONZERO_EXPONENT;
    }
    significand = significand >> hopper_to_fp32_shift;
    if (significand == 0) {
        return Gfloat(sign, HOPPER_ZERO_EXPONENT, 0);
    }
    return Gfloat(sign, exp, significand);
}


// Template helper function for matrix multiplication
template<typename T>
torch::Tensor Hopper_simulator::matrix_multiply_template(const torch::Tensor& A, const torch::Tensor& B, 
                                                        torch::ScalarType expected_dtype, 
                                                        const std::string& dtype_name,
                                                        Gfloat (*multiply_func)(uint16_t, uint16_t, int16_t),
                                                        int16_t zero_exponent) {
    // Validate input tensors have correct dtype
    if (A.dtype() != expected_dtype || B.dtype() != expected_dtype) {
        throw std::invalid_argument("Input tensors must be " + dtype_name + " dtype");
    }
    // Validate dimensions
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D");
    }
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);
    if (k != B.size(0)) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    // Check if B is column-major (Fortran contiguous)
    torch::Tensor B_col;
    bool is_col_major = (B.stride(0) == 1 && B.stride(1) == B.size(0));
    if (is_col_major) {
        B_col = B;
    } else {
        std::cout << "B is row-major replacing to col major" << std::endl;
        B_col = B.t().contiguous();
    }
    
    // Access tensor data
    auto A_data = A.data_ptr<T>();
    auto B_data = B_col.data_ptr<T>();
    
    // Initialize result tensor as float32
    auto result_tensor = torch::zeros({m, n}, torch::dtype(torch::kFloat32));
    auto result_accessor = result_tensor.accessor<float, 2>();
    
    // Determine number of threads
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    int total_elements = m * n;
    int elements_per_thread = total_elements / num_threads;
    int remaining_elements = total_elements % num_threads;
    
    auto compute_range = [&](int start_element, int end_element) {
        for (int element_idx = start_element; element_idx < end_element; ++element_idx) {
            int i = element_idx / n;
            int j = element_idx % n;
            std::vector<Gfloat> accumulator;
            accumulator.reserve(HOPPER_ACCUMULATOR_GROUP_SIZE);
            accumulator.push_back(Gfloat(false, HOPPER_ZERO_EXPONENT, 0));
            
            for (int l = 0; l < k; ++l) {
                uint16_t a_val = A_data[i * k + l].x;
                // B_col is always column-major, so access as B_col[j * k + l]
                uint16_t b_val = B_data[j * k + l].x;
                Gfloat product = multiply_func(a_val, b_val, zero_exponent);
                accumulator.push_back(product);
                
                if (accumulator.size() == HOPPER_ACCUMULATOR_GROUP_SIZE) {
                    Gfloat sum = group_sum(accumulator);
                    accumulator.clear();
                    accumulator.push_back(sum);
                }
            }
            
            if (!accumulator.empty()) {
                Gfloat final_result = group_sum(accumulator);
                float float_result = static_cast<float>(final_result);
                result_accessor[i][j] = float_result;
            } else {
                result_accessor[i][j] = 0.0f;
            }
        }
    };
    
    std::vector<std::thread> threads;
    int current_element = 0;
    for (int t = 0; t < num_threads; ++t) {
        int thread_elements = elements_per_thread + (t < remaining_elements ? 1 : 0);
        int end_element = current_element + thread_elements;
        threads.emplace_back(compute_range, current_element, end_element);
        current_element = end_element;
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return result_tensor;
}

// Matrix multiplication using bfloat16 to Gfloat conversion
torch::Tensor Hopper_simulator::matmul_bfloat16(const torch::Tensor& A, const torch::Tensor& B) {
    return matrix_multiply_template<c10::BFloat16>(A, B, torch::kBFloat16, "bfloat16", multiply_bfloat16_to_gfloat, HOPPER_ZERO_EXPONENT);
}

// Matrix multiplication using float16 to Gfloat conversion
torch::Tensor Hopper_simulator::matmul_float16(const torch::Tensor& A, const torch::Tensor& B) {
    return matrix_multiply_template<c10::Half>(A, B, torch::kFloat16, "float16", multiply_float16_to_gfloat, HOPPER_ZERO_EXPONENT);
}

// Unified matrix multiplication that dispatches based on tensor dtype
torch::Tensor Hopper_simulator::matmul(const torch::Tensor& A, const torch::Tensor& B) {
    // Check that both tensors have the same dtype
    if (A.dtype() != B.dtype()) {
        throw std::runtime_error("Both tensors must have the same data type");
    }
    
    // Dispatch based on tensor dtype
    if (A.dtype() == torch::kBFloat16) {
        return matmul_bfloat16(A, B);
    } else if (A.dtype() == torch::kFloat16) {
        return matmul_float16(A, B);
    } else {
        throw std::runtime_error("Unsupported tensor dtype. Only bfloat16 and float16 are supported");
    }
} 
