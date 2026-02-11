import os

from ..jax_compat import register_custom_call_target
from . import mpi_xla_bridge

# Import C++ module with FFI support (required for CPU)
from . import mpi_xla_bridge_cpu

try:
    from . import mpi_xla_bridge_cuda
except ImportError:
    HAS_CUDA_EXT = False
else:
    HAS_CUDA_EXT = True

try:
    from . import mpi_xla_bridge_cuda_cpp
except ImportError:
    HAS_CUDA_CPP_EXT = False
else:
    HAS_CUDA_CPP_EXT = True

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
mpi_xla_bridge_cpu.set_logging(_debug_enabled)
if HAS_CUDA_CPP_EXT:
    mpi_xla_bridge_cuda_cpp.set_logging(_debug_enabled)


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

# register custom call targets for CPU using C++ FFI implementation
for name in _ffi_primitives:
    register_custom_call_target(
        f"{name}_ffi",
        mpi_xla_bridge_cpu.ffi_targets[name],
        platform="cpu",
    )

if HAS_CUDA_EXT:
    for name, fn in mpi_xla_bridge_cuda.custom_call_targets.items():
        register_custom_call_target(name, fn, platform="CUDA", api_version=0)

# Register C++ FFI-based CUDA custom call targets
# These are used for primitives that have been ported to the new FFI API
if HAS_CUDA_CPP_EXT:
    _cuda_ffi_primitives = {
        "mpi_sendrecv",
    }
    for name in _cuda_ffi_primitives:
        if name in mpi_xla_bridge_cuda_cpp.ffi_targets:
            register_custom_call_target(
                f"{name}_ffi",
                mpi_xla_bridge_cuda_cpp.ffi_targets[name],
                platform="CUDA",
            )

if HAS_XPU_EXT:
    for name, fn in mpi_xla_bridge_xpu.custom_call_targets.items():
        register_custom_call_target(name, fn, platform="SYCL", api_version=0)
