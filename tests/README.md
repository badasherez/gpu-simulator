# GPU Simulator Tests

Bit-exact reproduction tests for GPU simulator. Supports Hopper (H100), Ampere (A100), and Custom configurations.

## Requirements

- PyTorch with CUDA support
- CuPy
- pytest
- NVIDIA GPU

## Running Tests

### Test with specific simulator

```bash
# Test Hopper simulator on H100
pytest tests/ --simulator=hopper --hardware=H100 -v -s

# Test Ampere simulator on A100
pytest tests/ --simulator=ampere --hardware=A100 -v -s

# Test Custom simulator on any hardware
pytest tests/ --simulator=custom --hardware=any -v -s
```

### Quick tests

```bash
# Default (Hopper on H100)
pytest tests/ -v -s

# Quick 10k tiles test
pytest tests/ -k "10k" --simulator=custom -v -s
```

### Run Specific Tests

```bash
# Quick test: 10,000 tiles
pytest tests/test_hopper_reproduction.py::TestBitExactTiles::test_10k_tiles_bitexact -v -s

# Full test: 100,000 tiles
pytest tests/test_hopper_reproduction.py::TestBitExactTiles::test_100k_tiles_bitexact -v -s

# Test specific dtype
pytest tests/ -v -s -k "bfloat16"
pytest tests/ -v -s -k "float16"
```

## Test Description

### test_10k_tiles_bitexact
- Tests 10,000 random 16×16 tiles
- Fast verification (~30 seconds on H100)
- Checks both bfloat16 and float16

### test_100k_tiles_bitexact
- Tests 100,000 random 16×16 tiles  
- Comprehensive verification (~5 minutes on H100)
- Checks both bfloat16 and float16

## Understanding Results

Tests verify **bit-exact reproduction** of tensor core operations:
- Simulator output must match GPU tensor core output bit-for-bit
- All 256 elements per tile must be identical (float32 precision)
- 0% tolerance - every single bit must match

### Expected Results on H100

✓ **100% bit-exact match** for individual 16×16 tiles

### Why Large Matrices Don't Match PyTorch

For matrices >64×64, PyTorch uses optimized kernels with different accumulation order:
- Individual tiles still match perfectly
- Large matrix results differ due to different tile accumulation order
- This is expected behavior - PyTorch changed their implementation in v2.x

## Build the C++ Extension

```bash
python3 setup.py build_ext --inplace
```

## Troubleshooting

**Import Error**: Make sure the C++ extension is built
```bash
cd /path/to/gpu-simulator
python3 setup.py build_ext --inplace
```

**CUDA Error**: Ensure CUDA is available
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**Wrong Hardware**: Use `--hardware=any` to skip hardware check
```bash
pytest tests/ --hardware=any -v -s
```

