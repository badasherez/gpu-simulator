"""
Exhaustive search for all computationally neutral subgroups.

Tests every subset of {-1, 0, 1, ..., 15} with at least 2 elements
and reports all subsets where test_computationally_neutral_subgroup returns True.

Index -1 represents the accumulator, indices 0-15 represent product positions.
"""

import torch
import sys
import time
from itertools import combinations

sys.path.insert(0, sys.path[0])  # ensure current dir is in path
from GPU_reproduction import test_computationally_neutral_subgroup, _build_nested_repr


def find_all_neutral_subgroups(dtype=torch.bfloat16):
    all_indices = list(range(-1, 16))  # {-1, 0, 1, ..., 15}
    n = len(all_indices)  # 17

    # Total subsets of size >= 2
    total_subsets = sum(1 for k in range(2, n + 1) for _ in combinations(all_indices, k))
    print(f"Testing {total_subsets:,} subsets of {{-1, 0, ..., 15}} with >= 2 elements")
    print(f"Dtype: {dtype}")
    print("=" * 70)

    neutral_subgroups = []
    tested = 0
    start_time = time.time()

    for size in range(2, n + 1):
        for subset in combinations(all_indices, size):
            S = list(subset)
            result = test_computationally_neutral_subgroup(S, dtype, turn_off_accumulator=False)
            tested += 1

            if result:
                neutral_subgroups.append(S)
                print(f"  ✓ NEUTRAL (size {size}): {S}")

            if tested % 5000 == 0:
                elapsed = time.time() - start_time
                rate = tested / elapsed if elapsed > 0 else 0
                remaining = (total_subsets - tested) / rate if rate > 0 else 0
                print(f"  Progress: {tested:,}/{total_subsets:,} "
                      f"({tested/total_subsets*100:.1f}%) - "
                      f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f" RESULTS")
    print("=" * 70)
    print(f"Total subsets tested: {tested:,}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Neutral subgroups found: {len(neutral_subgroups)}")
    print()

    if neutral_subgroups:
        # Group by size
        by_size = {}
        for s in neutral_subgroups:
            by_size.setdefault(len(s), []).append(s)

        for size in sorted(by_size.keys()):
            groups = by_size[size]
            print(f"Size {size} ({len(groups)} subgroups):")
            for g in groups:
                print(f"  {g}")
            print()

        # Nested tree representation
        print("Nested structure:")
        print(f"  {_build_nested_repr(neutral_subgroups)}")
        print()

    return neutral_subgroups


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    dtype = torch.bfloat16
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "float16":
            dtype = torch.float16

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    neutral = find_all_neutral_subgroups(dtype)
    print(f"\nDone. Found {len(neutral)} neutral subgroups total.")
