import torch
import gpu_simulator_py
from expirements.tensor_cores_mma import mma
from tqdm import tqdm
import struct
# Initialize both architectures
hopper = gpu_simulator_py.Hopper_simulator()
ampere = gpu_simulator_py.Ampere_simulator()

# Test with any supported precision
for dtype in [torch.bfloat16, torch.float16]:
    A = torch.randn(1024, 1024, dtype=dtype)
    B = torch.randn(1024, 1024, dtype=dtype)
    
    # Identical API - automatic dispatch!
    hopper_result = hopper.matmul(A, B)
    ampere_result = ampere.matmul(A, B)
    
    # Compare architectural differences
    precision_diff = torch.norm(hopper_result - ampere_result)
    print(f"{dtype} precision difference: {precision_diff:.6f}")


