"""
Bit-Exact Reproduction Tests for GPU Simulator.

Tests individual 16x16 tiles to verify bit-exact reproduction of 
tensor core operations. Supports Hopper, Ampere, and Custom simulators.

Usage:
    pytest tests/ --simulator=hopper --hardware=H100
    pytest tests/ --simulator=ampere --hardware=A100  
    pytest tests/ --simulator=custom --hardware=any
"""

import pytest
import torch
import struct


def float_to_bits(f):
    """Convert float32 to its 32-bit integer representation."""
    return struct.unpack('I', struct.pack('f', f))[0]


def bits_to_hex(bits, width=8):
    """Convert bits to hex string for debugging."""
    return f"0x{bits:0{width}x}"


class TestBitExactTiles:
    """
    Bit-exact tile reproduction tests.
    
    Tests individual 16x16 tiles against real GPU tensor core operations.
    """
    
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_100k_tiles_bitexact(self, simulator, mma_function, check_h100, dtype):
        """
        Test bit-exact reproduction over 100,000 random 16x16 tiles.
        
        Compares simulator output vs real GPU tensor core output.
        Every single bit must match.
        """
        num_tiles = 100_000
        total_elements = num_tiles * 256  # 256 elements per 16x16 tile
        
        mismatches = 0
        mismatch_examples = []
        
        print(f"\n{'='*60}")
        print(f"Testing {num_tiles:,} tiles ({total_elements:,} elements) - {dtype}")
        print(f"{'='*60}")
        
        for tile_idx in range(num_tiles):
            torch.manual_seed(tile_idx)
            torch.cuda.manual_seed(tile_idx)
            
            # Generate random 16x16 tiles
            A = torch.randn(1, 16, 16, dtype=dtype, device="cuda")
            B = torch.randn(1, 16, 16, dtype=dtype, device="cuda").mT
            
            # Run on real GPU tensor cores
            gpu_result = mma_function(A, B)
            
            # Run on simulator
            A_cpu = A[0].cpu()
            B_cpu = B[0].cpu().mT.contiguous().mT
            sim_result = simulator.matmul(A_cpu, B_cpu)
            
            # Bit-exact comparison (float32)
            sim_flat = sim_result.flatten()
            gpu_flat = gpu_result[0].flatten().cpu()
            
            for i in range(256):
                sim_bits = float_to_bits(sim_flat[i].item())
                gpu_bits = float_to_bits(gpu_flat[i].item())
                
                if sim_bits != gpu_bits:
                    mismatches += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append({
                            'tile': tile_idx,
                            'element': i,
                            'sim': sim_flat[i].item(),
                            'gpu': gpu_flat[i].item(),
                            'sim_bits': bits_to_hex(sim_bits),
                            'gpu_bits': bits_to_hex(gpu_bits),
                        })
            
            # Progress update every 10,000 tiles
            if (tile_idx + 1) % 10_000 == 0:
                print(f"  Processed {tile_idx + 1:,} tiles... (mismatches so far: {mismatches})")
        
        # Report results
        match_rate = (total_elements - mismatches) / total_elements * 100
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {dtype}")
        print(f"{'='*60}")
        print(f"Total elements: {total_elements:,}")
        print(f"Bit-exact matches: {total_elements - mismatches:,}")
        print(f"Mismatches: {mismatches}")
        print(f"Match rate: {match_rate:.6f}%")
        
        if mismatch_examples:
            print(f"\nFirst mismatches:")
            for m in mismatch_examples:
                print(f"  Tile {m['tile']}, elem {m['element']}: "
                      f"sim={m['sim']:.8e} ({m['sim_bits']}) vs "
                      f"gpu={m['gpu']:.8e} ({m['gpu_bits']})")
        
        print(f"{'='*60}\n")
        
        # Assert 100% bit-exact
        assert mismatches == 0, \
            f"FAILED: {mismatches:,} mismatches out of {total_elements:,} elements ({match_rate:.6f}% match rate)"
        
        print(f"✓ PASSED: All {total_elements:,} elements are bit-exact!")
    
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_10k_tiles_bitexact(self, simulator, mma_function, check_h100, dtype):
        """
        Quick test: 10,000 random 16x16 tiles for faster verification.
        """
        num_tiles = 10_000
        total_elements = num_tiles * 256
        
        mismatches = 0
        
        print(f"\n{'='*60}")
        print(f"Quick test: {num_tiles:,} tiles ({total_elements:,} elements) - {dtype}")
        print(f"{'='*60}")
        
        for tile_idx in range(num_tiles):
            torch.manual_seed(tile_idx)
            torch.cuda.manual_seed(tile_idx)
            
            A = torch.randn(1, 16, 16, dtype=dtype, device="cuda")
            B = torch.randn(1, 16, 16, dtype=dtype, device="cuda").mT
            
            gpu_result = mma_function(A, B)
            
            A_cpu = A[0].cpu()
            B_cpu = B[0].cpu().mT.contiguous().mT
            sim_result = simulator.matmul(A_cpu, B_cpu)
            
            sim_flat = sim_result.flatten()
            gpu_flat = gpu_result[0].flatten().cpu()
            
            for i in range(256):
                if float_to_bits(sim_flat[i].item()) != float_to_bits(gpu_flat[i].item()):
                    mismatches += 1
            
            if (tile_idx + 1) % 2_000 == 0:
                print(f"  Processed {tile_idx + 1:,} tiles... (mismatches so far: {mismatches})")
        
        match_rate = (total_elements - mismatches) / total_elements * 100
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {dtype}")
        print(f"{'='*60}")
        print(f"Total elements: {total_elements:,}")
        print(f"Bit-exact matches: {total_elements - mismatches:,}")
        print(f"Mismatches: {mismatches}")
        print(f"Match rate: {match_rate:.6f}%")
        print(f"{'='*60}\n")
        
        assert mismatches == 0, \
            f"FAILED: {mismatches:,} mismatches out of {total_elements:,} elements"
        
        print(f"✓ PASSED: All {total_elements:,} elements are bit-exact!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
