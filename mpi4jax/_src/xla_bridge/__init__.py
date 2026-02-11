import os

from ..jax_compat import register_custom_call_target
from . import mpi_xla_bridge
from . import mpi_xla_bridge_cpu

# Try to import pybind11-based C++ module with FFI support
try:
    from . import mpi_xla_bridge_cpu_cpp

    HAS_CPP_EXT = True
    # Check if FFI targets are available (new typed API)
    HAS_FFI_TARGETS = hasattr(mpi_xla_bridge_cpu_cpp, "ffi_targets")
except ImportError:
    HAS_CPP_EXT = False
    HAS_FFI_TARGETS = False

try:
    from . import mpi_xla_bridge_cuda
except ImportError:
    HAS_CUDA_EXT = False
else:
    HAS_CUDA_EXT = True

try:
    from . import mpi_xla_bridge_xpu  # noqa: F401
except ImportError:
    HAS_XPU_EXT = False
else:
    HAS_XPU_EXT = True

# setup logging


def _is_truthy(str_val):
    return str_val.lower() in ("true", "1", "on")


_debug_enabled = _is_truthy(os.getenv("MPI4JAX_DEBUG", ""))
mpi_xla_bridge.set_logging(_debug_enabled)
if HAS_CPP_EXT:
    mpi_xla_bridge_cpu_cpp.set_logging(_debug_enabled)


# Check if we should use FFI (new typed API) for C++ primitives
# Environment variable MPI4JAX_USE_FFI can be used to toggle (default: true)
_use_ffi = _is_truthy(os.getenv("MPI4JAX_USE_FFI", "true"))

# List of all primitives that have FFI implementations
_ffi_primitives = {
    "mpi_barrier",
    "mpi_allgather",
    "mpi_allreduce",
    "mpi_alltoall",
    "mpi_bcast",
    "mpi_gather",
    "mpi_scatter",
    "mpi_reduce",
    "mpi_scan",
    "mpi_send",
    "mpi_recv",
    "mpi_sendrecv",
}

# register custom call targets for CPU
# Use FFI API (api_version=1) for C++ primitives if available, otherwise legacy (api_version=0)
for name, fn in mpi_xla_bridge_cpu.custom_call_targets.items():
    # Use C++ FFI implementation if available and enabled
    if HAS_CPP_EXT and HAS_FFI_TARGETS and _use_ffi and name in _ffi_primitives:
        # Register with new FFI API (api_version=1 is default)
        # Use a different name for the FFI target to avoid conflicts
        register_custom_call_target(
            f"{name}_ffi",
            mpi_xla_bridge_cpu_cpp.ffi_targets[name],
            platform="cpu",  # api_version defaults to 1
        )
        # Also register legacy target for backward compatibility
        register_custom_call_target(name, fn, platform="cpu", api_version=0)
    else:
        register_custom_call_target(name, fn, platform="cpu", api_version=0)

if HAS_CUDA_EXT:
    for name, fn in mpi_xla_bridge_cuda.custom_call_targets.items():
        register_custom_call_target(name, fn, platform="CUDA", api_version=0)

if HAS_XPU_EXT:
    for name, fn in mpi_xla_bridge_xpu.custom_call_targets.items():
        register_custom_call_target(name, fn, platform="SYCL", api_version=0)
