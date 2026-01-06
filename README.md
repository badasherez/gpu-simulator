# Hawkeye: Reproducing GPU-Level Non-Determinism

A GPU simulator for reproducing and analyzing GPU-level non-deterministic behavior in tensor processing units. This research tool enables precise simulation of **any GPU architecture** by discovering and configuring hardware-specific characteristics.

## 🎯 How to Use

### Step 1: Characterize Your GPU

Run the characterization script on your hardware:

```bash
python3 expirements/GPU_reproduction.py [bfloat16|float16]
```

This discovers your GPU's characteristics:
- **Accumulation group size** (e.g., 17 for H100, 9 for A100)
- **Internal precision** in bits (e.g., 25 for H100, 24 for A100)  
- **Minimal max exponent** (e.g., -133 for H100, -132 for A100)

### Step 2: Configure Simulator

Update `utils/utils.h` with your GPU's values:

```cpp
// Your GPU constants
constexpr int16_t YOUR_GPU_ZERO_EXPONENT = -133;        // From minimal max exponent
constexpr int YOUR_GPU_SIGNIFICAND_WIDTH = 26;          // From internal precision + 1
constexpr int YOUR_GPU_ACCUMULATOR_GROUP_SIZE = 17;     // From group size
```

Then rebuild: `python3 setup.py build_ext --inplace`

## 🏗️ Pre-configured Architectures

- **NVIDIA Hopper (H100)**: 17-element groups, 25-bit precision
- **NVIDIA Ampere (A100)**: 9-element groups, 24-bit precision 


##  Clone and Build

```bash
git clone <repository-url>
cd gpu_simulator

# Build the C++ extension
python3 setup.py build_ext --inplace
```


## 📖 Usage Guide

```python
import torch
import gpu_simulator_py

# Pre-configured simulators
hopper = gpu_simulator_py.Hopper_simulator()  # H100 (pre-configured)
ampere = gpu_simulator_py.Ampere_simulator()  # A100 (pre-configured)
custom = gpu_simulator_py.Custom_simulator()  # YOUR GPU (after configuration)

# Configure for YOUR OWN GPU:
#
# 1. Characterize YOUR GPU:
#    $ python3 expirements/GPU_reproduction.py
#    Example output:
#      Accumulation group size... 13 elements
#      Internal precision... 23 bits
#      Minimal max exponent... -130
#
# 2. Edit utils/utils.h - Update CUSTOM_* constants:
#    constexpr int16_t CUSTOM_ZERO_EXPONENT = -130;           // Your min exponent
#    constexpr int CUSTOM_SIGNIFICAND_WIDTH = 24;             // Your precision + 1
#    constexpr int CUSTOM_ACCUMULATOR_GROUP_SIZE = 13;        // Your group size
#
# 3. Rebuild:
#    $ python3 setup.py build_ext --inplace
#
# 4. Use Custom_simulator for YOUR GPU:
import gpu_simulator_py
my_gpu = gpu_simulator_py.Custom_simulator()  # Simulates YOUR configured GPU
result = my_gpu.matmul(A, B)  # Bit-exact for YOUR hardware

# Create test matrices
A = torch.randn(128, 256, dtype=torch.bfloat16)
B = torch.randn(256, 128, dtype=torch.bfloat16)

# Unified API (recommended)
hopper_result = hopper.matmul(A, B) 
ampere_result = ampere.matmul(A, B)

```

## Reproduce Paper Results

```bash
python3 expirements/GPU_reproduction.py
```

## 🔧 Configuring for Your Own GPU

### Step 1: Characterize your hardware
```bash
python3 expirements/GPU_reproduction.py
```

Note these three values from the output:
- Accumulation group size
- Internal precision  
- Minimal max exponent

### Step 2: Update simulator constants

Edit `utils/utils.h` - Update the CUSTOM_* constants:
```cpp
constexpr int16_t CUSTOM_ZERO_EXPONENT = -130;           // Your minimal max exponent
constexpr int CUSTOM_SIGNIFICAND_WIDTH = 24;             // Your internal precision + 1  
constexpr int CUSTOM_ACCUMULATOR_GROUP_SIZE = 13;        // Your group size
```

### Step 3: Rebuild and test
```bash
python3 setup.py build_ext --inplace
pytest tests/ --simulator=custom --hardware=any -v -s
```

---

**Built for reproducible GPU research** 🔬 | **Enabling deterministic tensor analysis** 📊 | **Advancing ML system understanding** 🚀
