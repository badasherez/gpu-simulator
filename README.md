# Hawkeye: Reproducing GPU-Level Non-Determinism

A GPU simulator for reproducing and analyzing GPU-level non-deterministic behavior in tensor processing units. This research tool enables precise simulation of **any GPU architecture** by discovering and configuring hardware-specific characteristics.

Hawkeye is based on the paper "Hawkeye: Reproducing GPU-Level Non-Determinism" (https://arxiv.org/abs/2603.20421).

**Authors:**  
- Erez Badash
- [Dan Boneh](https://crypto.stanford.edu/~dabo/)  
- [Ilan Komargodski](https://www.cs.huji.ac.il/~ilank/)  
- [Megha Srivastava](https://meghabyte.github.io)
  
## 🏗️ Installation

```bash
git clone <repository-url>
cd gpu-simulator

pip install -r requirements.txt

# Build the C++ extension
python3 setup.py build_ext --inplace
```

## Quick Start

```python
import torch
import gpu_simulator_py

# Pre-configured simulators for known architectures
hopper = gpu_simulator_py.Hopper_simulator()  # H100
ampere = gpu_simulator_py.Ampere_simulator()  # A100

# Create test matrices (bfloat16 or float16)
A = torch.randn(128, 256, dtype=torch.bfloat16)
B = torch.randn(256, 128, dtype=torch.bfloat16)

# Simulate matrix multiplication — bit-exact with real hardware
hopper_result = hopper.matmul(A, B)
ampere_result = ampere.matmul(A, B)
```

### Pre-configured Architectures

| Architecture | GPU | Group Size | Internal Precision |
|---|---|---|---|
| **Hopper** | H100 | 17 elements | 25 bits |
| **Ampere** | A100 | 9 elements | 24 bits |

## 🔧 Configuring for Your Own GPU

### Step 1: Characterize your hardware

```bash
python3 experiments/GPU_reproduction.py [bfloat16|float16]
```

This runs an exhaustive characterization that discovers:
- **Accumulation group size and structure** (via neutral subgroup search)
- **Internal precision** in bits
- **Rounding modes** (shift and final summation)
- **Minimal max exponent**

### Step 2: Update simulator constants

The script outputs the exact values to configure. Edit `utils/utils.h`:

```cpp
constexpr int16_t CUSTOM_ZERO_EXPONENT = -132;           // Your minimal max exponent
constexpr int CUSTOM_SIGNIFICAND_WIDTH = 25;             // Your internal precision + 1
constexpr int CUSTOM_ACCUMULATOR_GROUP_SIZE = 9;         // Your group size
```

### Step 3: Rebuild and test

```bash
python3 setup.py build_ext --inplace
pytest tests/ --simulator=custom --hardware=any -v -s
```

### Step 4: Use your custom simulator

```python
import gpu_simulator_py

my_gpu = gpu_simulator_py.Custom_simulator()
result = my_gpu.matmul(A, B)  # Bit-exact for your hardware
```

## 📖 Running Tests

```bash
# Test Hopper simulator on H100
pytest tests/ --simulator=hopper --hardware=H100 -v -s

# Test Ampere simulator on A100
pytest tests/ --simulator=ampere --hardware=A100 -v -s

# Quick 10k tiles test with custom simulator
pytest tests/ -k "10k" --simulator=custom --hardware=any -v -s
```

## 📄 Reproduce Paper Results

```bash
python3 experiments/GPU_reproduction.py
```

