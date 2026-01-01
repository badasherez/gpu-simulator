# Hawkeye: Reproducing GPU-Level Non-Determinism


A  GPU simulator for reproducing and analyzing GPU level non-deterministic behavior in tensor processing units. This research tool enables precise simulation of **Hopper** and **Ampere** GPU architectures with their specific floating-point characteristics and accumulation patterns.

## 🎯 Research Overview

**Hawkeye** investigates the sources of non-determinism in modern GPU tensor cores by providing bit-accurate simulations of:

- **NVIDIA Hopper Architecture**: H100-class 
- **NVIDIA Ampere Architecture**: A100-class processing 


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

# Initialize simulators
hopper = gpu_simulator_py.Hopper_simulator()
ampere = gpu_simulator_py.Ampere_simulator()

# Create test matrices
A = torch.randn(128, 256, dtype=torch.bfloat16)
B = torch.randn(256, 128, dtype=torch.bfloat16)

# Unified API (recommended)
hopper_result = hopper.matmul(A, B)  # Auto-detects dtype
ampere_result = ampere.matmul(A, B)

```

## Reproduce Paper Results
```bash
python3 experiments/GPU_reproduction.py

```


---

**Built for reproducible GPU research** 🔬 | **Enabling deterministic tensor analysis** 📊 | **Advancing ML system understanding** 🚀
