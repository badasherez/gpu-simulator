#include "../include/Hopper_fp8_simulator.h"
#include <thread>

// Constructor
Hopper_fp8_simulator::Hopper_fp8_simulator() {}

// Group sum function — FP8 E4M3 uses 14-bit significand, single group
Gfloat Hopper_fp8_simulator::group_sum(const std::vector<Gfloat>& array) {
    // Shift from FP8 internal (14 bits) to FP32 significand (24 bits)
    constexpr int fp8_to_fp32_shift = HOPPER_FP8_SIGNIFICAND_WIDTH - FP32_SIGNIFICAND_WIDTH;
    // fp8_to_fp32_shift = 14 - 24 = -10  (internal is NARROWER than fp32)
    // So we shift RIGHT by |fp8_to_fp32_shift| = 10 when aligning to internal width,
    // and shift LEFT by 10 when expanding back to fp32.
    constexpr int internal_shift = FP32_SIGNIFICAND_WIDTH - HOPPER_FP8_SIGNIFICAND_WIDTH; // = 10
    
    // Step 1: Find the maximal exponent
    int16_t max_exponent = HOPPER_FP8_ZERO_EXPONENT;
    for (const auto& gfloat : array) {
        if (gfloat.get_exponent() > max_exponent) {
            max_exponent = gfloat.get_exponent();
        }
    }
    
    // Step 2: Align and accumulate all elements (towards_0 shift rounding = right shift truncation)
    int32_t significand = 0;
    for (const auto& gfloat : array) {
        int shift = max_exponent - gfloat.get_exponent();
        if (shift >= 32) {
            continue;
        }
        // Shift product significand to internal width, then align to max_exponent
        // Product significand is ~25 bits (from multiply). Shift right by internal_shift
        // to get to 14-bit internal width, then shift right by exponent difference.
        int32_t aligned = (gfloat.get_significand() >> internal_shift) >> shift;
        if (gfloat.get_sign()) {
            significand -= aligned;
        } else {
            significand += aligned;
        }
    }
    
    // Step 3: Normalize result
    bool sign = significand < 0;
    significand = std::abs(significand);
    int width = significand == 0 ? 0 : 32 - __builtin_clz(significand);
    if (width == 0) {
        return Gfloat(sign, HOPPER_FP8_ZERO_EXPONENT, 0);
    }
    
    int16_t exp = max_exponent + width - HOPPER_FP8_SIGNIFICAND_WIDTH;
    
    // Truncate to internal significand width (towards_0 output rounding)
    if (width > HOPPER_FP8_SIGNIFICAND_WIDTH) {
        significand = significand >> (width - HOPPER_FP8_SIGNIFICAND_WIDTH);
    } else {
        significand = significand << (HOPPER_FP8_SIGNIFICAND_WIDTH - width);
    }
    
    // Convert to FP32 output: expand significand to 24 bits
    if (exp < FP32_MIN_NONZERO_EXPONENT) {
        significand = significand >> (FP32_MIN_NONZERO_EXPONENT - exp);
        exp = FP32_MIN_NONZERO_EXPONENT;
    }
    // Shift from 14-bit internal to 24-bit fp32 significand
    significand = significand << internal_shift;
    if (significand == 0) {
        return Gfloat(sign, HOPPER_FP8_ZERO_EXPONENT, 0);
    }
    return Gfloat(sign, exp, significand);
}

// Matrix multiplication: A [M,K] e4m3 @ B^T [N,K] e4m3 → D [M,N] f32
torch::Tensor Hopper_fp8_simulator::matmul(const torch::Tensor& A, const torch::Tensor& B) {
    if (A.scalar_type() != torch::kFloat8_e4m3fn || B.scalar_type() != torch::kFloat8_e4m3fn) {
        throw std::invalid_argument("Input tensors must be float8_e4m3fn dtype");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D");
    }
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);  // B is [N, K] (transposed layout, like WGMMA)
    if (K != B.size(1)) {
        throw std::invalid_argument("K dimensions don't match: A is [M,K], B must be [N,K]");
    }
    
    // Access raw bytes (fp8 = 1 byte per element)
    auto A_data = reinterpret_cast<const uint8_t*>(A.data_ptr());
    auto B_data = reinterpret_cast<const uint8_t*>(B.data_ptr());
    
    // Result tensor in float32
    auto result_tensor = torch::zeros({M, N}, torch::dtype(torch::kFloat32));
    auto result_accessor = result_tensor.accessor<float, 2>();
    
    // Multithreaded computation
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    int total_elements = M * N;
    int elements_per_thread = total_elements / num_threads;
    int remaining_elements = total_elements % num_threads;
    
    auto compute_range = [&](int start_element, int end_element) {
        for (int element_idx = start_element; element_idx < end_element; ++element_idx) {
            int i = element_idx / N;
            int j = element_idx % N;
            
            // Single accumulation group: acc + all K products
            std::vector<Gfloat> accumulator;
            accumulator.reserve(HOPPER_FP8_ACCUMULATOR_GROUP_SIZE);
            // Accumulator starts at zero (acc position = beginning of group)
            accumulator.push_back(Gfloat(false, HOPPER_FP8_ZERO_EXPONENT, 0));
            
            // D[i,j] = sum_k A[i,k] * B[j,k]  (B is [N,K] transposed)
            for (int l = 0; l < K; ++l) {
                uint8_t a_val = A_data[i * K + l];
                uint8_t b_val = B_data[j * K + l];
                Gfloat product = multiply_fp8e4m3_to_gfloat(a_val, b_val, HOPPER_FP8_ZERO_EXPONENT);
                accumulator.push_back(product);
            }
            
            // Single group_sum for all products (no sub-grouping)
            Gfloat final_result = group_sum(accumulator);
            result_accessor[i][j] = static_cast<float>(final_result);
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

