import torch
import math
from tqdm import tqdm
from tensor_cores_mma import mma_fp8_e4m3

# WGMMA dimensions for fp8 e4m3
K = 32   # reduction dimension
M = 64   # output rows (WGMMA fixed)
N = 128  # output cols (WGMMA fixed)

# Chosen so sqrt values are exactly representable in e4m3fn:
#   sqrt(V_large) = 2^7  = 128   (normal in e4m3, e=14 m=0)
#   sqrt(V_small) = 2^-8 = 1/256 (subnormal in e4m3, e=0 m=2)
#   2*sqrt(V_large) = 2^8 = 256  (normal in e4m3, e=15 m=0, not NaN since m!=7)
V_LARGE = 2.0 ** 14
V_SMALL = 2.0 ** (-16)


def find_internal_precision_fp8():
    """
    Find the internal precision of the FP8 E4M3 WGMMA accumulator.

    Sets up: D[0,0] = 1*1 + 1*(-1) + 2^upper * 2^floor = 2^(-c)

    We increment c until the result deviates from 2^(-c).

    e4m3fn smallest representable power-of-2 is 2^(-9),
    so the maximum testable c is 18 (both inputs = 2^(-9), product = 2^(-18)).

    Returns:
        int or None: number of precision bits, or None if e4m3 limit reached
    """
    MIN_E4M3_EXP = -9

    left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")

    left[0, 0, 0] = 1.0
    right[0, 0, 0] = 1.0
    left[0, 0, 1] = 1.0
    right[0, 0, 1] = -1.0

    c = 1
    while True:
        upper = math.ceil(-c / 2)
        floor = math.floor(-c / 2)

        if upper < MIN_E4M3_EXP or floor < MIN_E4M3_EXP:
            return None

        left[0, 0, 2] = 2.0 ** upper
        right[0, 0, 2] = 2.0 ** floor

        result = mma_fp8_e4m3(left, right)
        if result[0, 0, 0].item() != 2.0 ** (-c):
            return c - 1
        c += 1


def find_accumulator_precision_fp8():
    """
    Find the internal precision of the f32 accumulator input in WGMMA.

    Sets up: D[0,0] = 1*1 + 1*(-1) + acc
                     = 0 + acc
                     = acc   where acc = 2^(-c)

    Product 0 and 1 cancel (1 + (-1) = 0). The accumulator holds
    the tiny value 2^(-c). We increment c until the result deviates.

    Since the accumulator is f32, it can represent 2^(-149) (subnormal).
    If this test gives the same precision as find_internal_precision_fp8,
    the accumulator is truncated to the same internal representation.
    If it gives more bits, the accumulator has a wider path.

    Returns:
        int: number of precision bits
    """
    left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")

    left[0, 0, 0] = 1.0
    right[0, 0, 0] = 1.0
    left[0, 0, 1] = 1.0
    right[0, 0, 1] = -1.0

    c = 1
    while c <= 24:  # f32 has 23-bit mantissa, so 24 is beyond its limit
        acc = torch.zeros(1, M, N, dtype=torch.float32, device="cuda")
        acc[0, 0, 0] = 2.0 ** (-c)

        result = mma_fp8_e4m3(left, right, acc=acc)
        if result[0, 0, 0].item() != 2.0 ** (-c):
            return c - 1
        c += 1
    return None  # full f32 precision


def find_output_precision_fp8(c):
    """
    Find the output (final) precision of the WGMMA accumulator.

    The internal precision is c bits. This test checks whether the
    output has additional bits beyond c.

    Sets up:
      product 0: left[0]*right[0] = 2^(-c)          (tiny, at the internal limit)
      products 1..2^d: left[i]*right[i] = 1.0 each   (sum = 2^d)

    D[0,0] = 2^(-c) + 2^d

    Representing this exactly requires c+d significand bits.
    We increment d until the result deviates. Output precision = c + d_max.

    Max testable d is 4 (2^4 = 16 products of 1.0, plus the tiny product = 17 slots ≤ 32).

    Args:
        c: internal precision in bits (from find_internal_precision_fp8)

    Returns:
        int: total output significand width in bits (c + 1 + extra carry bits)
    """
    upper = math.ceil(-c / 2)
    floor = math.floor(-c / 2)

    last_pass = -1
    for d in range(0, 6):  # d = 0,1,2,3,4,5
        num_ones = 2 ** d
        if num_ones + 1 > K:  # need num_ones + 1 product slots
            break

        left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
        right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")

        # Product 0: 2^(-c)
        left[0, 0, 0] = 2.0 ** upper
        right[0, 0, 0] = 2.0 ** floor

        # Products 1..num_ones: each = 1.0, sum = 2^d
        for i in range(1, num_ones + 1):
            left[0, 0, i] = 1.0
            right[0, 0, i] = 1.0

        result = mma_fp8_e4m3(left, right)
        expected = float(num_ones) + 2.0 ** (-c)
        actual = result[0, 0, 0].item()

        if actual != expected:
            # 2^d + 2^(-c) needs (c + d + 1) significand bits
            # last passing d means output has (c + last_pass + 1) sig bits
            return c + last_pass + 1
        last_pass = d

    return c + last_pass + 1  # all tested d values worked


def test_accumulator_max_exponent_fp8():
    """
    Test whether the WGMMA accumulator preserves a value at the
    maximum f32 exponent (2^127) when all products are zero.

    D = 0 + acc, where acc = 2^127 (largest power of 2 in f32).

    Returns:
        bool: True if 2^127 passes through unchanged
    """
    left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")
    acc = torch.zeros(1, M, N, dtype=torch.float32, device="cuda")
    acc[0, 0, 0] = 2.0 ** 127

    result = mma_fp8_e4m3(left, right, acc=acc)
    return result[0, 0, 0].item() == 2.0 ** 127


def test_final_output_rounding_fp8(output_prec):
    """
    Detect the rounding mode of the final WGMMA output using candidate
    elimination (Algorithm 9 adapted for FP8 E4M3).

    The output has output_prec significand bits (e.g. 14). We create a
    product whose exponent makes the ULP = 4 at that precision, then
    probe with +3 and -1 corrections.

    Product = (1.5 * 2^e)^2 = 2.25 * 2^(2e) where 2e = output_prec,
    so e = output_prec // 2.  Exponent of product = 2e + 1 = output_prec + 1.
    ULP at output_prec sig bits = 2^(output_prec + 1 - (output_prec - 1)) = 4.

    Returns:
        tuple: (rounding_mode, reduction_amount)
    """
    candidates = [
        "nearest_tie_towards_0",
        "nearest_tie_away_from_0",
        "nearest_tie_towards_even",
        "towards_0",
        "away_from_0",
        "towards_even"
    ]

    e = output_prec // 2                  # exponent for each factor
    large_val = 1.5 * (2.0 ** e)          # e.g. 1.5 * 2^7 = 192 for output_prec=14
    ref = 2.25 * (2.0 ** output_prec)     # = large_val^2, exactly representable
    # ULP at output_prec significand bits = 4
    # next representable above ref: ref + 4
    # next representable below ref: ref - 4

    # 1. Test 1 (baseline): internal sum = ref + 3
    left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")
    left[0, 0, 0] = large_val / 2
    right[0, 0, 0] = large_val             # product 0 = ref
    left[0, 0, 1] = large_val / 2
    right[0, 0, 1] = large_val  
    left[0, 0, 2] = 3.0
    right[0, 0, 2] = 1.0                   # product 1 = 3

    D = mma_fp8_e4m3(left, right)

    # 6. Test 1: internal sum was ref + 3.
    #    At output_prec sig bits, ULP = 4. Nearest representable above ref is ref + 4.
    #    ref + 3 is distance 3 from ref and distance 1 from ref + 4.
    #    If baseline == ref: rounded towards zero (not nearest) → eliminates nearest + away_from_0
    #    If baseline == ref + 4: rounded to nearest / away from zero → eliminates towards_0
    baseline = D[0, 0, 0].item()
    if baseline == ref:
        candidates = [m for m in candidates if "nearest" not in m]
        candidates = [m for m in candidates if m != "away_from_0"]
    elif baseline == ref + 4:
        candidates = [m for m in candidates if m != "towards_0"]

    # 7. Test 2 (sign-changing correction): internal sum = ref - 1
    #    ref - 1 is distance 1 from ref and distance 3 from ref - 4.
    left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")
    left[0, 0, 0] = large_val / 2
    right[0, 0, 0] = large_val             # product 0 = ref
    left[0, 0, 1] = large_val / 2
    right[0, 0, 1] = large_val  
    left[0, 0, 2] = 1.0
    right[0, 0, 2] = -1.0                  # product 1 = -1

    D = mma_fp8_e4m3(left, right)

    actual = D[0, 0, 0].item()
    reduction = baseline - actual

    # Test 2 analysis:
    #   towards_0:    ref - 1 → ref - 4 (truncate towards zero for positive), reduction > 0
    #   towards_even: ref - 1 → ref (even neighbor), reduction = 0
    #   away_from_0:  ref - 1 → ref (away from zero = up for positive), reduction = 0
    #   nearest modes: ref - 1 → ref (distance 1), reduction = 0
    if reduction > 0:
        candidates = [m for m in candidates if m not in ("towards_even", "away_from_0")]
        candidates = [m for m in candidates if "nearest" not in m]
    elif reduction == 0:
        candidates = [m for m in candidates if m != "towards_0"]

    rounding_mode = candidates[0] if len(candidates) == 1 else candidates
    return rounding_mode, int(reduction)


def test_subnormal_normalization_fp8(internal_size):
    """
    Test whether subnormal e4m3 inputs are normalized before multiplication.

    e4m3fn subnormals:
      e=0, value = 2^(-6) * (m/8), m=1..7
      Min normal exponent = -6 (bias=7, emin=1-7=-6)

    Sets up:
      product 0: 2^(-7) * 2^6 = 2^(-1) = 0.5       (left is SUBNORMAL)
      product 1: 2^(-7) * (-2^6) = -0.5              (cancels product 0)
      product 2: 2^upper * 2^floor = 2^(-1-internal_size)  (tiny residual)

    D[0,0] should equal 2^(-1-internal_size).

    If the subnormal 2^(-7) is normalized before multiplying (treated as
    1.0 * 2^(-7) with full significand), the products have enough precision
    for exact cancellation and the residual survives.
    If NOT normalized, the subnormal's reduced significand may corrupt the result.

    Returns:
        bool: True if subnormal normalization occurs (result is correct)
    """
    sub_normal_exp = -6  # e4m3fn min normal exponent

    left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")

    # Product 0: subnormal * normal = 2^(-1)
    left[0, 0, 0] = 2.0 ** (sub_normal_exp - 1)   # 2^(-7), subnormal in e4m3
    right[0, 0, 0] = 2.0 ** (-sub_normal_exp)      # 2^6 = 64, normal in e4m3

    # Product 1: subnormal * (-normal) = -2^(-1)
    left[0, 0, 1] = 2.0 ** (sub_normal_exp - 1)    # 2^(-7), subnormal
    right[0, 0, 1] = -(2.0 ** (-sub_normal_exp))   # -64

    # Product 2: tiny residual = 2^(-1-internal_size)
    upper = math.ceil((-1 - internal_size) / 2)
    floor = math.floor((-1 - internal_size) / 2)
    left[0, 0, 2] = 2.0 ** upper
    right[0, 0, 2] = 2.0 ** floor

    result = mma_fp8_e4m3(left, right)

    return result[0, 0, 0].item() == 2.0 ** (-1 - internal_size)


def test_normalization_overflow_fp8(internal_size):
    """
    Test whether normalization overflow occurs in the internal accumulator.

    Sets up: D[0,0] = 1.5*1.5 + 1.5*(-1.5) + 2^upper * 2^floor
                     = 2.25 - 2.25 + 2^(-internal_size)
                     = 2^(-internal_size)

    The 1.5 values produce products with mantissa bits that may cause
    overflow during normalization of the internal representation.
    If the result differs from 2^(-internal_size), normalization overflow occurs.

    Returns:
        bool: True if normalization overflow occurs
    """
    left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")
    left[0, 0, 0] = 1.5
    right[0, 0, 0] = 1.5
    left[0, 0, 1] = 1.5
    right[0, 0, 1] = -1.5
    upper = math.ceil(-internal_size / 2)
    floor = math.floor(-internal_size / 2)
    left[0, 0, 2] = 2.0 ** upper
    right[0, 0, 2] = 2.0 ** floor
    result = mma_fp8_e4m3(left, right)

    return result[0, 0, 0].item() != 2.0 ** (-internal_size)


def detect_rounding_mode_fp8(c):
    """
    Detect the rounding mode used during shift operations in summation.

    WGMMA convention: D[m,n] = sum_k left[m,k] * right[n,k]

    Args:
        c: Internal representation size in bits (from find_internal_precision_fp8)

    Returns:
        str or list: The detected rounding mode(s)
    """
    candidates = [
        "nearest_tie_towards_0",
        "nearest_tie_away_from_0",
        "nearest_tie_towards_even",
        "towards_0",
        "away_from_0",
        "towards_even"
    ]

    exp_a = -math.ceil((c + 1) / 2)
    exp_b = -math.floor((c + 1) / 2)

    left = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    right = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")
    left[0, 0, 0] = 1.0
    left[0, 0, 1] = 1.0
    left[0, 0, 2] = 2.0 ** exp_a
    right[0, 0, 0] = 1.0
    right[0, 0, 1] = -1.0
    right[0, 0, 2] = 2.0 ** exp_b

    # Test 1: positive half-ULP — D = 1 - 1 + 2^(-(c+1))
    # result == 0: rounded towards 0 (eliminates away_from_0)
    # result != 0: rounded away from 0
    if mma_fp8_e4m3(left, right)[0, 0, 0].item() == 0.0:
        candidates = [m for m in candidates if "away_from_0" not in m]
    else:
        candidates = ["nearest_tie_away_from_0", "away_from_0"]

    # Test 2: negative half-ULP — D = 1 - 1 - 2^(-(c+1))
    # Symmetric: result == 0 means towards 0, result != 0 means away from 0
    left[0, 0, 2] = -(2.0 ** exp_a)
    if mma_fp8_e4m3(left, right)[0, 0, 0].item() == 0.0:
        candidates = [m for m in candidates if "away_from_0" not in m]
    else:
        candidates = [m for m in candidates if "away_from_0" in m]

    # Test 3: product with 1.5 * 2^exp to test tie-to-even
    exp_base = -math.ceil(c / 2)
    exp_b2 = -math.floor(c / 2)
    left[0, 0, 2] = 2.0 ** exp_base + 2.0 ** (exp_base - 1)
    right[0, 0, 2] = 2.0 ** exp_b2

    if mma_fp8_e4m3(left, right)[0, 0, 0].item() == 2.0 ** (-c):
        candidates = [m for m in candidates if "even" not in m]
    else:
        candidates = [m for m in candidates if "even" in m]

    # Test 4
    left[0, 0, 2] = 2.0 ** exp_a
    right[0, 0, 2] = 2.0 ** exp_b + 2.0 ** (exp_b - 1)

    if mma_fp8_e4m3(left, right)[0, 0, 0].item() == 0.0:
        candidates = [m for m in candidates if "towards_0" in m]
    else:
        candidates = [m for m in candidates if "towards_0" not in m]

    # Test 5: disambiguate nearest_tie_towards_0 vs towards_0
    if "nearest_tie_towards_0" in candidates and "towards_0" in candidates:
        left[0, 0, 0] = 1.0
        left[0, 0, 1] = 1.0
        left[0, 0, 2] = 3.0 * 2.0 ** exp_a
        right[0, 0, 0] = 1.0
        right[0, 0, 1] = -1.0
        right[0, 0, 2] = 2.0 ** (exp_b - 1)

        if mma_fp8_e4m3(left, right)[0, 0, 0].item() == 0.0:
            candidates = ["towards_0"]
        else:
            candidates = ["nearest_tie_towards_0"]

    return candidates[0] if len(candidates) == 1 else candidates


def test_computationally_neutral_subgroup_fp8(S):
    """
    Test for a Computationally Neutral Subgroup in FP8 E4M3 WGMMA.

    WGMMA convention: D[m,n] = sum_k left[m,k] * right[n,k] + acc[m,n]

    Index -1 represents the accumulator. Indices 0..31 are product indices.

    Sets up two scenarios for output element D[0,0]:
      1) Cancellation: elements in S have large values that sum to zero,
         elements outside S have small values.
      2) Zeroed: elements in S are zero, elements outside S have same small values.

    If D[0,0] is bitwise equal in both scenarios, S is computationally neutral.

    Args:
        S: A set of indices from {-1, 0, 1, ..., 31}.
           -1 = accumulator, 0..31 = product indices.
           Must have len >= 2.

    Returns:
        bool: True if the subgroup is computationally neutral
    """
    S_set = set(S)
    sqrt_large = math.sqrt(V_LARGE)  # 2^7  = 128
    sqrt_small = math.sqrt(V_SMALL)  # 2^-8
    acc_in_S = -1 in S_set

    # ── Test 1: Cancellation Scenario ──
    A_cancel = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    B_cancel = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")

    for k in range(K):
        if k in S_set:
            A_cancel[0, 0, k] = sqrt_large
            B_cancel[0, 0, k] = sqrt_large
        else:
            A_cancel[0, 0, k] = sqrt_small
            B_cancel[0, 0, k] = sqrt_small

    # Accumulator value for cancel test
    acc_cancel = torch.zeros(1, M, N, dtype=torch.float32, device="cuda")
    acc_cancel[0, 0, 0] = V_LARGE if acc_in_S else V_SMALL

    # Create cancellation within S: negate half, sum to zero
    S_list = sorted(S)
    n = len(S_list)
    neg_count = (n + 1) // 2

    for i in range(neg_count):
        idx = S_list[i]
        if idx == -1:
            acc_cancel[0, 0, 0] *= -1.0
        else:
            A_cancel[0, 0, idx] = -sqrt_large

    # For odd n: double one positive element so the sum is exactly zero
    if n % 2 == 1:
        idx = S_list[neg_count]  # first positive element
        if idx == -1:
            acc_cancel[0, 0, 0] *= 2.0
        else:
            A_cancel[0, 0, idx] = 2.0 * sqrt_large

    D_cancel = mma_fp8_e4m3(A_cancel, B_cancel, acc=acc_cancel)
    R_cancel = D_cancel[0, 0, 0].item()

    # ── Test 2: Zeroed Subgroup Scenario (Baseline) ──
    A_zero = torch.zeros(1, M, K, dtype=torch.float8_e4m3fn, device="cuda")
    B_zero = torch.zeros(1, N, K, dtype=torch.float8_e4m3fn, device="cuda")

    for k in range(K):
        if k in S_set:
            A_zero[0, 0, k] = 0.0
            B_zero[0, 0, k] = 0.0
        else:
            A_zero[0, 0, k] = sqrt_small
            B_zero[0, 0, k] = sqrt_small

    acc_zero = torch.zeros(1, M, N, dtype=torch.float32, device="cuda")
    acc_zero[0, 0, 0] = 0.0 if acc_in_S else V_SMALL

    D_zero = mma_fp8_e4m3(A_zero, B_zero, acc=acc_zero)
    R_zero = D_zero[0, 0, 0].item()

    return R_cancel == R_zero


def find_accumulation_group_size_fp8():
    """
    Find accumulation group structure by searching all contiguous subgroups
    of {0, 1, ..., 31}, each with and without the accumulator (index -1).

    Contiguous product subgroups: 496.
    Each tested with and without accumulator: 496 * 2 = 992 tests.

    Returns:
        list: All neutral subgroups found (each is a list that may contain -1).
    """
    neutral_subgroups = []

    # Generate all contiguous subgroups of {0,...,31} with size >= 2
    # Then for each, also create a variant that includes the accumulator (-1)
    all_subsets = []
    for start in range(K):
        for end in range(start + 2, K + 1):
            products = list(range(start, end))
            all_subsets.append(products)                  # without accumulator
            all_subsets.append([-1] + products)           # with accumulator

    # Also test accumulator + single product (size 2)
    for k in range(K):
        all_subsets.append([-1, k])

    print(f"  Testing {len(all_subsets)} subgroups "
          f"(contiguous products ± accumulator)...")

    for subset in tqdm(all_subsets, desc="  Searching neutral subgroups"):
        if test_computationally_neutral_subgroup_fp8(subset):
            neutral_subgroups.append(subset)

    # Print results
    if neutral_subgroups:
        nested = _build_nested_repr(neutral_subgroups)
        print(f"  Nested structure: {nested}")
    else:
        print("  No neutral subgroups found!")
        print("  (All 32 products + accumulator form one group)")

    return neutral_subgroups


def _build_nested_repr(neutral_subgroups):
    """
    Build a nested tuple representation showing how contiguous subgroups
    nest inside each other.

    E.g. if {0,...,7} and {0,...,15} are neutral within {0,...,31}:
        ((0, 1, 2, 3, 4, 5, 6, 7), 8, 9, 10, 11, 12, 13, 14, 15), 16, ..., 31)
    """
    if not neutral_subgroups:
        return "()"

    sorted_groups = sorted(neutral_subgroups, key=len, reverse=True)

    def nest(group, subgroups):
        group_set = set(group)

        proper_subs = [sg for sg in subgroups if set(sg) < group_set]

        # Keep only maximal subgroups
        maximal_subs = []
        for sg in proper_subs:
            sg_set = set(sg)
            if not any(sg_set < set(other) for other in proper_subs):
                maximal_subs.append(sg)

        def fmt(x):
            return "acc" if x == -1 else str(x)

        if not maximal_subs:
            return "(" + ", ".join(fmt(x) for x in sorted(group)) + ")"

        covered = set()
        nested_parts = []
        for sg in sorted(maximal_subs, key=lambda s: min(s)):
            inner_subs = [s for s in proper_subs if set(s) < set(sg)]
            nested_parts.append((min(sg), nest(sg, inner_subs)))
            covered.update(sg)

        all_parts = []
        for elem in sorted(group):
            if elem in covered:
                for min_val, repr_str in nested_parts:
                    if elem == min_val:
                        all_parts.append(repr_str)
                        break
            else:
                all_parts.append(fmt(elem))

        return "(" + ", ".join(all_parts) + ")"

    root = sorted_groups[0]
    return nest(root, sorted_groups[1:])


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {device_name}")
    print(f"Testing: FP8 E4M3 via WGMMA (m64n128k32)")
    print(f"V_large = 2^14, V_small = 2^(-16)")
    print("=" * 70)

    # Step 1: Internal precision
    print("\nFinding internal precision...")
    precision = find_internal_precision_fp8()
    if precision is None:
        print(f"  >= 18 bits (reached e4m3 input limit)\n")
    else:
        print(f"  {precision} bits\n")

    # Step 2: Accumulator precision
    print("Finding accumulator input precision...")
    acc_precision = find_accumulator_precision_fp8()
    if acc_precision is None:
        print(f"  >= 24 bits (full f32)\n")
    else:
        print(f"  {acc_precision} bits\n")

    # Step 3: Output precision
    output_prec = None
    if precision is not None:
        print("Finding output precision...")
        output_prec = find_output_precision_fp8(precision)
        extra = output_prec - precision - 1  # subtract 1 for the implicit bit
        print(f"  {output_prec} significand bits ({precision} fractional + 1 implicit + {extra} carry)\n")

    # Step 4: Accumulator max exponent passthrough
    print("Testing accumulator max exponent (2^127)...")
    acc_max_exp = test_accumulator_max_exponent_fp8()
    print(f"  {'PASS (2^127 preserved)' if acc_max_exp else 'FAIL (2^127 corrupted)'}\n")

    # Step 4: Normalization overflow
    overflow = None
    if precision is not None:
        print("Testing normalization overflow...")
        overflow = test_normalization_overflow_fp8(precision)
        print(f"  {'YES' if overflow else 'NO'}\n")

    # Step 4: Subnormal normalization
    subnormal_norm = None
    if precision is not None:
        print("Testing subnormal normalization...")
        subnormal_norm = test_subnormal_normalization_fp8(precision)
        print(f"  {'YES' if subnormal_norm else 'NO'}\n")

    # Step 7: Shift rounding mode
    rounding = None
    if precision is not None:
        print("Detecting shift rounding mode...")
        rounding = detect_rounding_mode_fp8(precision)
        if rounding is None:
            print("  Could not determine\n")
        elif isinstance(rounding, list):
            print(f"  Candidates: {rounding}\n")
        else:
            print(f"  {rounding}\n")

    # Step 8: Output rounding mode
    final_rounding = None
    if output_prec is not None:
        print("Detecting output rounding mode...")
        final_rounding, final_reduction = test_final_output_rounding_fp8(output_prec)
        if isinstance(final_rounding, list):
            print(f"  Candidates: {final_rounding} (reduction={final_reduction})\n")
        else:
            print(f"  {final_rounding} (reduction={final_reduction})\n")

    # Step 9: Find accumulation group structure (contiguous subgroups only)
    print("Finding accumulation group structure (contiguous subgroups)...")
    neutral = find_accumulation_group_size_fp8()

    # Summary
    print("\n" + "=" * 70)
    print(" FP8 E4M3 CHARACTERIZATION RESULTS")
    print("=" * 70)
    print(f"GPU:           {device_name}")
    print(f"Instruction:   QGMMA.64x128x32.F32.E4M3.E4M3")
    print(f"K dimension:   {K}")
    if precision is None:
        print(f"Internal precision: >= 18 bits (e4m3 input limit)")
    else:
        print(f"Internal precision: {precision} bits")
    if acc_precision is None:
        print(f"Accum precision:    >= 24 bits (full f32)")
    else:
        print(f"Accum precision:    {acc_precision} bits")
    if output_prec is not None:
        extra = output_prec - precision - 1
        print(f"Output precision:   {output_prec} significand bits ({precision} frac + 1 implicit + {extra} carry)")
    print(f"Acc max exp (2^127):{'PASS' if acc_max_exp else 'FAIL'}")
    if overflow is not None:
        print(f"Norm overflow:      {'YES' if overflow else 'NO'}")
    if subnormal_norm is not None:
        print(f"Subnormal norm:     {'YES' if subnormal_norm else 'NO'}")
    if rounding is not None:
        mode_str = rounding if isinstance(rounding, str) else str(rounding)
        print(f"Shift rounding:     {mode_str}")
    if final_rounding is not None:
        mode_str2 = final_rounding if isinstance(final_rounding, str) else str(final_rounding)
        print(f"Output rounding:    {mode_str2}")
    if neutral:
        smallest = min(neutral, key=len)
        print(f"Smallest neutral group: size {len(smallest)} "
              f"(indices {smallest[0]}..{smallest[-1]})")
        sizes = sorted(set(len(sg) for sg in neutral))
        print(f"All neutral sizes: {sizes}")
    else:
        print("No neutral contiguous subgroups found")
    print("=" * 70 + "\n")

