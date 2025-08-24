import torch
import math
from tensor_cores_mma import mma

def reproduction_internal_representaion(dtype):
    left = torch.zeros(1,16, 16, dtype=dtype, device="cuda")
    right = torch.zeros(1,16, 16, dtype=dtype, device="cuda").mT
    left[0, 0, 0] = 1.0
    right[0, 0, 0] = 1.0
    left[0, 0, 1] = 1.0
    right[0, 1, 0] = -1.0
    c = 1
    while True:
        upper = math.ceil(- c/2)
        floor = math.floor(- c/2)
        left[0, 0, 2] = 2.0 ** upper
        right[0, 2, 0] = 2.0 ** floor
        result = mma(left, right)
        if result[0, 0, 0] != 2.0**(-c):
            print(f"Internal representation size is {c-1}")
            break
        c += 1
def recovery_normalization_overflow(internal_size, dtype):
    left = torch.zeros(1,16, 16, dtype=dtype, device="cuda")
    right = torch.zeros(1,16, 16, dtype=dtype, device="cuda").mT
    left[0, 0, 0] = 1.5
    right[0, 0, 0] = 1.5
    left[0, 0, 1] = 1.5
    right[0, 1, 0] = -1.5
    upper = math.ceil(- internal_size/2)
    floor = math.floor(- internal_size/2)
    left[0, 0, 2] = 2.0 ** upper
    right[0, 2, 0] = 2.0 ** floor
    result = mma(left, right)
    if result[0, 0, 0] != 2.0**(-internal_size):
        print(f"Normalization overflow occured ")
    else:
        print(f"No Normalization overflow occured")

def recovery_normalization_subnormal_multiplication(internal_size, dtype):
    left = torch.zeros(1,16, 16, dtype=dtype, device="cuda")
    right = torch.zeros(1,16, 16, dtype=dtype, device="cuda").mT
    if dtype == torch.float16:
        sub_normal_exp = -14
    else:
        sub_normal_exp = -126
    left[0, 0, 0] = 2.0 ** (sub_normal_exp - 1)
    right[0, 0, 0] = 2.0 ** (-sub_normal_exp)
    left[0, 0, 1] = 2.0 ** (sub_normal_exp - 1)
    right[0, 1, 0] = -2.0 ** (-sub_normal_exp)
    upper = math.ceil((-1 -internal_size)/2)
    floor = math.floor((-1 -internal_size)/2)
    left[0, 0, 2] = 2.0 ** upper
    right[0, 2, 0] = 2.0 ** floor
    result = mma(left, right)
    if result[0, 0, 0] != 2.0**(-1-internal_size):
        print(f"No normalization after subnormal multiplication occured - subnormal multiplications have smaller mantissa")
    else:
        print(f"Normalization after subnormal multiplication occured")

def test_computationally_neutral_subgroup(S, dtype=torch.bfloat16):
    """
    Test for a Computationally Neutral Subgroup
    
    Args:
        S: A set of product indices {k1, k2, ..., km} from {-1,0, ..., 16}
        dtype: Data type for tensor operations (torch.bfloat16 or torch.float16)
    
    Returns:
        bool: True if the subgroup is computationally neutral (bitwise equal results)
    """
    # Constants
    V_large = 2.0 ** 20
    V_small = 2.0 ** (-20)
    
    # Test 1: Cancellation Scenario
    # Initialize matrices for cancellation test
    A_cancel = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    B_cancel = torch.zeros(1, 16, 16, dtype=dtype, device="cuda").mT
    C_cancel = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    
    # Set C[0,0] based on whether 0 is in S
    C_cancel[0, 0, 0] = V_large if -1 in S else V_small
    
    # Set up product values for cancellation scenario
    for k in range(16): 
        if k not in S:
            A_cancel[0, 0, k] = math.sqrt(V_small)
            B_cancel[0, k, 0] = math.sqrt(V_small)
        else:
            A_cancel[0, 0, k] = math.sqrt(V_large)
            B_cancel[0, k, 0] = math.sqrt(V_large)
    
    # Multiply half of the elements in S by -1 to get zero sum
    S_list = list(S)
    half_size = len(S_list) // 2
    for i in range(half_size):
        k = S_list[i]
        if k != -1:  # Don't modify k=0 as it affects C matrix
            A_cancel[0, 0, k] *= -1
    
    if len(S_list) % 2 == 1:
        k = S_list[half_size -1]
        A_cancel[0, 0, k] *= A_cancel[0, 0, k] * 2.0
    
    
    # Compute result for cancellation scenario
    D_cancel = mma(left = A_cancel, right = B_cancel, acc = C_cancel)
    R_cancel = D_cancel[0, 0, 0].item()
    
    # Test 2: Zeroed Subgroup Scenario (Baseline)
    # Initialize matrices for zero test
    A_zero = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    B_zero = torch.zeros(1, 16, 16, dtype=dtype, device="cuda").mT
    C_zero = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    
    # Set C[0,0] based on whether 0 is in S
    C_zero[0, 0, 0] = 0.0 if -1 in S else V_small
    
    # Set up product values for zero scenario
    for k in range(16): 
        if k not in S:
            A_zero[0, 0, k] = math.sqrt(V_small)
            B_zero[0, k, 0] = math.sqrt(V_small)
        else:
            # Elements in S are set to zero
            A_zero[0, 0, k] = 0.0
            B_zero[0, k, 0] = 0.0
    
    # Compute result for zero scenario
    D_zero =  mma(left = A_zero, right = B_zero, acc = C_zero)
    R_zero = D_zero[0, 0, 0].item()
    
    return R_cancel == R_zero


if __name__ == "__main__":
    reproduction_internal_representaion(torch.bfloat16)
    recovery_normalization_overflow(24, torch.bfloat16)
    recovery_normalization_subnormal_multiplication(24, torch.float16)
