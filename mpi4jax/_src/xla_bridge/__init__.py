import os

from ..jax_compat import register_custom_call_target

# Import C++ module with FFI support (required for CPU)
from . import mpi_xla_bridge_cpu

# Re-export MPI_STATUS_IGNORE_ADDR from C++ module
MPI_STATUS_IGNORE_ADDR = mpi_xla_bridge_cpu.MPI_STATUS_IGNORE_ADDR

# Try to import CUDA C++ extension
try:
    from . import mpi_xla_bridge_cuda
except ImportError:
    HAS_CUDA_EXT = False
else:
    HAS_CUDA_EXT = True

# Try to import XPU C++ extension
try:
    from . import mpi_xla_bridge_xpu
except ImportError:
    HAS_XPU_EXT = False
else:
    HAS_XPU_EXT = True

# setup logging


def _is_truthy(str_val):
    return str_val.lower() in ("true", "1", "on")


_debug_enabled = _is_truthy(os.getenv("MPI4JAX_DEBUG", ""))
mpi_xla_bridge_cpu.set_logging(_debug_enabled)


# List of all MPI primitives using FFI
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

# Register FFI targets for CPU
for name in _ffi_primitives:
    register_custom_call_target(
        f"{name}_ffi",
        mpi_xla_bridge_cpu.ffi_targets[name],
        platform="cpu",
    )

# Register FFI targets for CUDA
if HAS_CUDA_EXT:
    for name in _ffi_primitives:
        if name in mpi_xla_bridge_cuda.ffi_targets:
            register_custom_call_target(
                f"{name}_ffi",
                mpi_xla_bridge_cuda.ffi_targets[name],
                platform="CUDA",
            )

# Register FFI targets for XPU/SYCL
if HAS_XPU_EXT:
    for name in _ffi_primitives:
        if name in mpi_xla_bridge_xpu.ffi_targets:
            register_custom_call_target(
                f"{name}_ffi",
                mpi_xla_bridge_xpu.ffi_targets[name],
                platform="SYCL",
            )
