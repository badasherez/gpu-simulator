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
            return c-1
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
    
    return result[0, 0, 0] != 2.0**(-internal_size)

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
    
    return result[0, 0, 0] == 2.0**(-1-internal_size)

def test_computationally_neutral_subgroup(S, dtype=torch.bfloat16, turn_of_accumulator =True):
    """
    Test for a Computationally Neutral Subgroup
    
    Args:
        S: A set of product indices {k1, k2, ..., km} from {-1,0, ..., 15}
        dtype: Data type for tensor operations (torch.bfloat16 or torch.float16)
        turn_of_accumulator: ignore the accumulator when testing for computational neutrality
    
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
    if turn_of_accumulator:
        C_cancel[0, 0, 0] = 0
    
    # Set up product values for cancellation scenario
    for k in range(16): 
        if k not in S:
            A_cancel[0, 0, k] = math.sqrt(V_small)
            B_cancel[0, k, 0] = math.sqrt(V_small)
        else:
            A_cancel[0, 0, k] = math.sqrt(V_large)
            B_cancel[0, k, 0] = math.sqrt(V_large)
    
    # Create cancellation: half positive, half negative
    S_list = list(S)
    n = len(S_list)
    
    # Negate ceil(n/2) elements to create larger negative group
    neg_count = (n + 1) // 2  # Ceiling division
    for i in range(neg_count):
        k = S_list[i]
        if k != -1:
            A_cancel[0, 0, k] *= -1
        else:
            C_cancel[0, 0, 0] *= -1.0
    
    # For odd n: multiply one element in the smaller (positive) group by 2
    # This makes: -ceil(n/2)*V + (floor(n/2)-1)*V + 2*V = 0
    if n % 2 == 1:
        # Multiply the first element of the positive group by 2
        k = S_list[neg_count]  # First element after negated ones
        if k != -1:
            A_cancel[0, 0, k] *= 2.0
        else:
            C_cancel[0, 0, 0] *= 2.0
    
    
    # Compute result for cancellation scenario
    D_cancel = mma(left = A_cancel, right = B_cancel, acc = C_cancel)
    R_cancel = D_cancel[0, 0, 0].item()
    
    # Test 2: Zeroed Subgroup Scenario (Baseline)
    # Initialize matrices for zero test
    A_zero = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    B_zero = torch.zeros(1, 16, 16, dtype=dtype, device="cuda").mT
    C_zero = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    
    # Set C[0,0] based on whether -1 is in S
    C_zero[0, 0, 0] = 0.0 if -1 in S else V_small
    if turn_of_accumulator:
        C_zero[0, 0, 0] = 0.0
    
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

def detect_rounding_mode(c, dtype=torch.bfloat16):
    """
    Detect the rounding mode used during shift operations in summation.
    
    Args:
        c: Internal representation size in bits
        dtype: Data type to test
    
    Returns:
        str: The detected rounding mode
    """
    candidates = [
        "nearest_tie_towards_0",
        "nearest_tie_towards_minus_infinity",
        "nearest_tie_towards_even",
        "towards_0",
        "towards_minus_infinity",
        "towards_even"
    ]
    
    exp_a = -math.ceil((c + 1) / 2)
    exp_b = -math.floor((c + 1) / 2)
    
    left = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    right = torch.zeros(1, 16, 16, dtype=dtype, device="cuda").mT
    left[0, 0, 0] = 1.0
    left[0, 0, 1] = 1.0
    left[0, 0, 2] = 2.0 ** exp_a
    right[0, 0, 0] = 1.0
    right[0, 1, 0] = -1.0
    right[0, 2, 0] = 2.0 ** exp_b
    
    # Test 1
    if mma(left, right)[0, 0, 0].item() != 0.0:
        return None
    
    # Test 2
    left[0, 0, 2] = -(2.0 ** exp_a)
    if mma(left, right)[0, 0, 0].item() == 0.0:
        candidates = [m for m in candidates if "minus_infinity" not in m]
    else:
        candidates = ["nearest_tie_towards_minus_infinity", "towards_minus_infinity"]
    
    # Test 3
    exp_base = -math.ceil(c / 2)
    exp_b2 = -math.floor(c / 2)
    left[0, 0, 2] = 2.0 ** exp_base + 2.0 ** (exp_base - 1)
    right[0, 2, 0] = 2.0 ** exp_b2
    
    if mma(left, right)[0, 0, 0].item() == 2.0 ** (-c):
        candidates = [m for m in candidates if "even" not in m]
    else:
        candidates = [m for m in candidates if "even" in m]
    
    # Test 4
    left[0, 0, 2] = 2.0 ** exp_a
    right[0, 2, 0] = 2.0 ** exp_b + 2.0 ** (exp_b - 1)
    
    if mma(left, right)[0, 0, 0].item() == 0.0:
        candidates = [m for m in candidates if "towards_0" in m]
    else:
        candidates = [m for m in candidates if "towards_0" not in m]
    
    # Test 5
    if "nearest_tie_towards_0" in candidates and "towards_0" in candidates:
        left[0, 0, 0] = 1.0
        left[0, 0, 1] = 1.0
        left[0, 0, 2] = 3.0 * 2.0 ** exp_a
        right[0, 0, 0] = 1.0
        right[0, 1, 0] = -1.0
        right[0, 2, 0] = 2.0 ** (exp_b - 1)
        
        if mma(left, right)[0, 0, 0].item() == 0.0:
            candidates = ["towards_0"]
        else:
            candidates = ["nearest_tie_towards_0"]
    
    return candidates[0] if len(candidates) == 1 else candidates

def test_final_summation_rounding(dtype=torch.bfloat16):
    """
    Determine the rounding mode applied to the final summation result.
    
    Returns:
        tuple: (rounding_mode, reduction_amount)
    """
    left = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    right = torch.zeros(1, 16, 16, dtype=dtype, device="cuda").mT
    
    # Test 1: Baseline
    left[0, 0, 0] = 1.5 * (2.0 ** 12)
    right[0, 0, 0] = 1.5 * (2.0 ** 12)
    left[0, 0, 1] = 3.0
    right[0, 1, 0] = 1.0
    
    baseline = mma(left, right)[0, 0, 0].item()
    
    # Test 2: With -1 correction
    left[0, 0, 1] = 1.0
    right[0, 1, 0] = -1.0
    
    actual = mma(left, right)[0, 0, 0].item()
    reduction = baseline - actual
    
    # Determine mode
    if reduction == 0:
        rounding_mode = "round_to_nearest"
    elif reduction > 0:
        rounding_mode = "truncation"
    else:
        rounding_mode = "unknown"
    
    return rounding_mode, int(reduction)

def find_minimal_max_exponent(m, dtype=torch.bfloat16):
    """
    Find the minimal max exponent during accumulation.
    
    Discovers the smallest exponent value the GPU can represent during
    internal accumulation, revealing the dynamic range of the accumulator.
    
    Args:
        m: GPU internal representation size (from test 2)
        dtype: Data type to test
    
    Returns:
        int: Minimal max exponent value
    """
    A = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    B = torch.zeros(1, 16, 16, dtype=dtype, device="cuda").mT
    C = torch.zeros(1, 16, 16, dtype=torch.float32, device="cuda")
    
    for e in range(0, -200, -1):  # Test negative exponents
        # Set A_{i,1} = 2^⌈e/2⌉, B_{1,j} = 2^⌊e/2⌋
        A[0, 0, 0] = 2.0 ** math.ceil(e / 2)
        B[0, 0, 0] = 2.0 ** math.floor(e / 2)
        
        # Set A_{i,2} = 2^⌈(e-m)/2⌉, B_{2,j} = 2^⌊(e-m)/2⌋
        A[0, 0, 1] = 2.0 ** math.ceil((e - m) / 2)
        B[0, 1, 0] = -2.0 ** math.floor((e - m) / 2)
        
        # Compute
        D = mma(A, B, acc=C)
        result = D[0, 0, 0].item()
        expected = 2.0 ** e
        
        if result == expected:
            # Found the limit
            return e + 1
    
    return None  # Not found

def test_extended_range_accumulation(dtype=torch.bfloat16):
    """
    Test temporary extended-range accumulation (bfloat16 only).
    
    Verifies intermediate accumulation can exceed fp32 range before normalizing.
    
    Returns:
        bool: True if test passes
    """
    A = torch.zeros(1, 16, 16, dtype=dtype, device="cuda")
    B = torch.zeros(1, 16, 16, dtype=dtype, device="cuda").mT
    C = torch.zeros(1, 16, 16, dtype=torch.float32, device="cuda")
    
    A[0, 0, 0] = 2.0 ** 127
    A[0, 0, 1] = 2.0 ** 127
    A[0, 0, 2] = 2.0 ** 127
    
    B[0, 0, 0] = 2.0
    B[0, 1, 0] = -2.0
    B[0, 2, 0] = 1.0
    
    result = mma(A, B, acc=C)
    return result[0, 0, 0].item() == 2.0 ** 127

def test_accumulator_position(dtype=torch.bfloat16, num_products = 16):
    """
    Determine if accumulator is added at the beginning or end of the group.
    
    Tests if the accumulator (C matrix) is computationally neutral.
    - If neutral: accumulator is treated like products (at the end)
    - If not neutral: accumulator is special (at the beginning)
    
    Args:
        dtype: Data type to test
    
    Returns:
        str: "beginning" or "end"
    """
    # Test with S = {-1} union with product indices
    S = {-1} | set(range(num_products))
    
    is_neutral = test_computationally_neutral_subgroup(S, dtype, turn_of_accumulator = False)
    if is_neutral:
        return "beginning"
    else:
        return "end"

def find_accumulation_group_size(dtype=torch.bfloat16):
    """
    Find the accumulation group size for the GPU architecture.
    
    Returns the number of elements (products + accumulator) in one group.
    - Hopper (H100): 17 elements
    - Ampere (A100): 9 elements
    
    Returns:
        int: Accumulation group size (including accumulator)
    """
    k = 2
    while k < 16:
        S = [i for i in range(k)]
        if test_computationally_neutral_subgroup(S, dtype):
            return k
        k *= 2
    return 16



if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("GPU Tensor Core Characterization")
        print("="*70)
        print("\nUsage:")
        print("  python3 GPU_reproduction.py [dtype]")
        print("\nOptions:")
        print("  bfloat16    Test with bfloat16 (default)")
        print("  float16     Test with float16")
        print("\nExamples:")
        print("  python3 GPU_reproduction.py")
        print("  python3 GPU_reproduction.py bfloat16")
        print("  python3 GPU_reproduction.py float16")
        print("\nThis script characterizes GPU tensor core behavior:")
        print("  - Accumulation group size")
        print("  - Internal representation precision")
        print("  - Rounding modes")
        print("  - Normalization behavior")
        exit(0)
    
    dtype = torch.bfloat16
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "float16":
            dtype = torch.float16
        elif sys.argv[1].lower() == "bfloat16":
            dtype = torch.bfloat16
        else:
            print(f"Unknown dtype: {sys.argv[1]}")
            print(f"Use: bfloat16 or float16")
            exit(1)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    
    print(f"\nGPU: {device_name}")
    print(f"Testing dtype: {dtype}")
    print("="*70)
    
    # 1. Accumulation group size
    print("Accumulation group size...")
    num_products = find_accumulation_group_size(dtype)
    print(f"{num_products} elements\n")
    
    # 2. Accumulator position
    print("Accumulator position...")
    acc_position = test_accumulator_position(dtype, num_products)
    print(f"{acc_position}\n")
    
    # 3. Internal representation
    print("Internal representation...")
    c = reproduction_internal_representaion(dtype)
    if c is None:
        print("ERROR: Could not determine")
        exit(1)
    print(f"{c} bits\n")
    
    # 4. Shift operation rounding
    print("Shift operation rounding...")
    shift_rounding = detect_rounding_mode(c, dtype)
    mode_str = shift_rounding if isinstance(shift_rounding, str) else shift_rounding[0]
    print(f"{mode_str}\n")
    
    # 5. Normalization overflow  
    print("Normalization overflow...")
    overflow = recovery_normalization_overflow(c, dtype)
    print(f"{'YES' if overflow else 'NO'}\n")
    
    # 6. Final summation rounding
    print("Final summation rounding...")
    final_rounding, reduction = test_final_summation_rounding(dtype)
    print(f"{final_rounding}\n")
    
    # 7. Subnormal behavior
    print("Subnormal multiplication...")
    subnormal_norm = recovery_normalization_subnormal_multiplication(c, torch.float16)
    print(f"{'YES' if subnormal_norm else 'NO'}\n")
    
    # 8. Extended-range accumulation (bfloat16 only)
    if dtype == torch.bfloat16:
        print("Extended-range accumulation...")
        extended_range = test_extended_range_accumulation(dtype)
        print(f"{'SUPPORTED' if extended_range else 'NOT SUPPORTED'}\n")
    else:
        extended_range = None
    
    # 9. Minimal max exponent (bfloat16 only)
    if dtype == torch.bfloat16:
        print("Minimal max exponent...")
        min_max_exp = find_minimal_max_exponent(c, dtype)
        if min_max_exp:
            print(f"{min_max_exp}\n")
        else:
            print("Could not determine\n")
    else:
        min_max_exp = None
    
    # Summary
    print("\n" + "="*70)
    print(" CHARACTERIZATION RESULTS")
    print("="*70)
    print(f"GPU:                      {device_name}")
    print(f"Dtype:                    {dtype}")
    
    # Group size with clarification
    if acc_position == "beginning":
        group_size = num_products + 1  # group_size includes accumulator
        print(f"Accumulation:             {num_products} products + accumulator (at beginning)")
        print(f"Group size:               {group_size} elements")
    else:
        print(f"Accumulation:             Accumulator at end - NOT IMPLEMENTED")
        # print(f"Group size:               {group_size} elements (not supported)")
    
    print(f"Internal precision:       {c} bits")
    print(f"Shift rounding:           {mode_str}")
    print(f"Normalization overflow:   {'YES' if overflow else 'NO'}")
    if dtype == torch.bfloat16:
        print(f"Extended range:           {'SUPPORTED' if extended_range else 'NOT SUPPORTED'}")
    print(f"Final rounding:           {final_rounding}")
    print(f"Subnormal normalization:  {'YES' if subnormal_norm else 'NO'}")
    if dtype == torch.bfloat16 and min_max_exp:
        print(f"Minimal max exponent:     {min_max_exp}")
    
    # Simulator compatibility check
    print("\n" + "="*70)
    print(" SIMULATOR COMPATIBILITY")
    print("="*70)
    
    compatible = True
    warnings = []
    
    # Check critical assumptions
    if acc_position != "beginning":
        compatible = False
        warnings.append("✗ Accumulator position: Simulator assumes 'beginning', got '{}'".format(acc_position))
    
    if mode_str != "towards_0":
        compatible = False
        warnings.append("✗ Shift rounding: Simulator assumes 'towards_0', got '{}'".format(mode_str))
    
    if final_rounding != "truncation":
        compatible = False
        warnings.append("✗ Final rounding: Simulator assumes 'truncation', got '{}'".format(final_rounding))
    
    if overflow:
        compatible = False
        warnings.append("✗ Normalization overflow: Simulator assumes 'NO', got 'YES'")
    
    # Warnings for edge cases
    if dtype == torch.bfloat16:
        if not extended_range:
            warnings.append("⚠ Extended range not supported - may give wrong results for extreme values")
    
    if subnormal_norm:
        warnings.append("⚠ Subnormal normalized - simulator assumes no normalization, may give wrong results")
    
    if compatible and not warnings:
        print("✓ SIMULATOR COMPATIBLE")
        print("  Your GPU matches simulator assumptions.")
        print("  You can use Custom_simulator with your configuration:")
        print(f"\n  Configure in utils/utils.h:")
        print(f"    CUSTOM_ACCUMULATOR_GROUP_SIZE = {group_size}")
        print(f"    CUSTOM_SIGNIFICAND_WIDTH = {c + 1}")
        if dtype == torch.bfloat16 and min_max_exp:
            print(f"    CUSTOM_ZERO_EXPONENT = {min_max_exp}")
    elif compatible:
        print("✓ SIMULATOR COMPATIBLE (with warnings)")
        print("  Configure in utils/utils.h:")
        print(f"    CUSTOM_ACCUMULATOR_GROUP_SIZE = {group_size}")
        print(f"    CUSTOM_SIGNIFICAND_WIDTH = {c + 1}")
        if dtype == torch.bfloat16 and min_max_exp:
            print(f"    CUSTOM_ZERO_EXPONENT = {min_max_exp}")
        print("\n  Warnings:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("✗ SIMULATOR NOT COMPATIBLE")
        print("  Your GPU has characteristics not implemented in the simulator:")
        for w in warnings:
            print(f"  {w}")
        print("\n  The simulator will NOT produce correct results for your GPU.")
    
    print("="*70 + "\n")
