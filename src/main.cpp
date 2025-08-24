#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include "gfloat.h"
#include "../utils/utils.h"
#include "Hopper_simulator.h"
#include <torch/torch.h>

// Function to create random Gfloat matrix
std::vector<std::vector<Gfloat>> create_random_matrix(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<std::vector<Gfloat>> matrix(rows, std::vector<Gfloat>(cols));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = dis(gen);
            // Convert to Gfloat format (simplified)
            bool sign = val < 0;
            int16_t exp = 0;  // Simple exponent
            int32_t sig = static_cast<int32_t>(std::abs(val) * 1000);
            matrix[i][j] = Gfloat(sign, exp, sig);
        }
    }
    
    return matrix;
}

// Function to multiply two Gfloat matrices
std::vector<std::vector<Gfloat>> multiply_matrices(
    const std::vector<std::vector<Gfloat>>& A,
    const std::vector<std::vector<Gfloat>>& B) {
    
    int m = A.size();
    int k = A[0].size();
    int n = B[0].size();
    
    std::vector<std::vector<Gfloat>> result(m, std::vector<Gfloat>(n));
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Gfloat sum(false, HOPPER_ZERO_EXPONENT, 0);  // Initialize to zero
            
            for (int l = 0; l < k; ++l) {
                Gfloat product = A[i][l] * B[l][j];
                // Simple addition (you might want to use group_sum here)
                sum = sum + product;
            }
            
            result[i][j] = sum;
        }
    }
    
    return result;
}

int main() {
    std::cout << "Hopper GPU Simulator - Matrix Multiplication Benchmark Comparison" << std::endl;
    std::cout << "=============================================================" << std::endl;
    
    const int size = 4096;
    
    // Create random bfloat16 tensors
    std::cout << "\nCreating " << size << "x" << size << " bfloat16 tensors..." << std::endl;
    auto A = torch::randn({size, size}, torch::dtype(torch::kBFloat16));
    auto B = torch::randn({size, size}, torch::dtype(torch::kBFloat16));
    
    std::cout << "Matrix A: " << A.sizes()[0] << "x" << A.sizes()[1] << std::endl;
    std::cout << "Matrix B: " << B.sizes()[0] << "x" << B.sizes()[1] << std::endl;
    
    // Create Hopper simulator instance
    Hopper_simulator simulator;
    
    // Test matrix multiplication function
    std::cout << "\nTesting matrix multiplication function..." << std::endl;
    
    // Function: matrix_multiply_direct_bfloat16
    std::cout << "\n=== matrix_multiply_direct_bfloat16 ===" << std::endl;
    std::vector<double> times;
    const int num_runs = 3;
    B = B.t().contiguous().t();
    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = simulator.matrix_multiply_direct_bfloat16(A, B);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double seconds = duration.count() / 1000.0;
        times.push_back(seconds);
        
        std::cout << "Run " << (run + 1) << ": " << seconds << " seconds" << std::endl;
    }
    
    // Calculate statistics
    double avg_time = 0.0;
    for (double t : times) {
        avg_time += t;
    }
    avg_time /= times.size();
    
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());
    
    // Print results
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\nMatrix multiplication (direct_bfloat16):" << std::endl;
    std::cout << "  Average time: " << avg_time << " seconds" << std::endl;
    std::cout << "  Min time: " << min_time << " seconds" << std::endl;
    std::cout << "  Max time: " << max_time << " seconds" << std::endl;
    
    // Calculate FLOPS
    double flops = 2.0 * size * size * size;
    double gflops = flops / (avg_time * 1e9);
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
    
    std::cout << "\nBenchmark completed successfully!" << std::endl;
    
    return 0;
}
