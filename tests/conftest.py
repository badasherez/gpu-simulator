"""
Pytest configuration and fixtures for GPU simulator tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "expirements"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--hardware",
        action="store",
        default="H100",
        help="Hardware to test on: H100, A100, or any"
    )
    parser.addoption(
        "--simulator",
        action="store",
        default="hopper",
        help="Simulator to test: hopper, ampere, or custom"
    )


@pytest.fixture(scope="session")
def hardware(request):
    """Get the hardware requirement from command line."""
    return request.config.getoption("--hardware")


@pytest.fixture(scope="session")
def check_cuda():
    """Verify CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture(scope="session")
def check_h100(check_cuda, hardware):
    """Verify running on H100 (Hopper architecture)."""
    device_name = torch.cuda.get_device_name(0)
    
    if hardware.upper() == "H100":
        if "H100" not in device_name and "Hopper" not in device_name.lower():
            pytest.skip(f"Test requires H100. Found: {device_name}")
    elif hardware.upper() == "A100":
        if "A100" not in device_name and "Ampere" not in device_name.lower():
            pytest.skip(f"Test requires A100. Found: {device_name}")
    elif hardware.upper() != "ANY":
        pytest.skip(f"Unknown hardware option: {hardware}")
    
    return device_name


@pytest.fixture(scope="session")
def simulator(request):
    """Initialize the selected simulator."""
    import gpu_simulator_py
    
    sim_type = request.config.getoption("--simulator").lower()
    
    if sim_type == "hopper":
        return gpu_simulator_py.Hopper_simulator()
    elif sim_type == "ampere":
        return gpu_simulator_py.Ampere_simulator()
    elif sim_type == "custom":
        return gpu_simulator_py.Custom_simulator()
    else:
        raise ValueError(f"Unknown simulator: {sim_type}. Use: hopper, ampere, or custom")


@pytest.fixture(scope="session")
def hopper_simulator():
    """Initialize Hopper simulator (deprecated - use 'simulator' fixture)."""
    import gpu_simulator_py
    return gpu_simulator_py.Hopper_simulator()


@pytest.fixture(scope="session")
def ampere_simulator():
    """Initialize Ampere simulator (deprecated - use 'simulator' fixture)."""
    import gpu_simulator_py
    return gpu_simulator_py.Ampere_simulator()


@pytest.fixture(scope="session")
def mma_function(check_cuda):
    """Get the real GPU MMA function."""
    from tensor_cores_mma import mma
    return mma


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed
