import os

from mpi4py import MPI as _MPI

from ..jax_compat import register_custom_call_target

# Import C++ module with FFI support (required for CPU)
from . import mpi_xla_bridge_cpu

# Re-export MPI_STATUS_IGNORE_ADDR from C++ module
MPI_STATUS_IGNORE_ADDR = mpi_xla_bridge_cpu.MPI_STATUS_IGNORE_ADDR


# ============================================================================
# MPI ABI compatibility check
# ============================================================================
# Check that the MPI library used at runtime matches what we built against.
# Mismatches can cause silent corruption since MPI handle types differ:
# - OpenMPI: handles are pointers (8 bytes on 64-bit)
# - MPICH: handles are signed 32-bit integers (4 bytes)


def _check_mpi_abi_compatibility():
    """Check if runtime MPI is compatible with build-time MPI."""
    build_abi = mpi_xla_bridge_cpu.MPI_ABI_INFO

    # Get runtime MPI info
    runtime_comm_handle = _MPI._handleof(_MPI.COMM_WORLD)
    # Convert to signed int64 like we do in to_mpi_handle
    if runtime_comm_handle >= (1 << 63):
        runtime_comm_handle = runtime_comm_handle - (1 << 64)

    runtime_lib_version = _MPI.Get_library_version()

    # Check 1: Compare COMM_WORLD handles
    # If they differ significantly, the MPI implementations are different
    build_comm_handle = build_abi["comm_world_handle"]

    # Check 3: Library version string prefix
    build_lib_version = build_abi["mpi_library_version"]

    # Determine if we're dealing with pointer-style (OpenMPI) or int-style (MPICH) handles
    def _is_pointer_style(handle):
        # Pointers on 64-bit are typically > 2^32 and positive
        # MPICH handles are small integers (often negative or small positive)
        return handle > (1 << 32) or handle < -(1 << 30)

    build_is_pointer = _is_pointer_style(build_comm_handle)
    runtime_is_pointer = _is_pointer_style(runtime_comm_handle)

    errors = []

    # Critical: handle type mismatch (pointer vs int)
    if build_is_pointer != runtime_is_pointer:
        errors.append(
            f"MPI handle type mismatch: mpi4jax was built with "
            f"{'pointer-style (OpenMPI)' if build_is_pointer else 'integer-style (MPICH)'} "
            f"handles but runtime MPI uses "
            f"{'pointer-style (OpenMPI)' if runtime_is_pointer else 'integer-style (MPICH)'} "
            f"handles."
        )

    # Warning: library version mismatch (could be compatible but worth noting)
    # Extract vendor name from version string
    build_vendor = build_lib_version.split()[0:2]  # e.g., ["Open", "MPI"] or ["MPICH"]
    runtime_vendor = runtime_lib_version.split()[0:2]

    if build_vendor != runtime_vendor:
        errors.append(
            f"MPI library mismatch: mpi4jax was built with "
            f"'{' '.join(build_vendor)}' but runtime uses "
            f"'{' '.join(runtime_vendor)}'."
        )

    if errors:
        error_msg = (
            "MPI ABI incompatibility detected!\n"
            + "\n".join(f"  - {e}" for e in errors)
            + f"\n\nBuild-time MPI: {build_lib_version[:60]}..."
            + f"\nRuntime MPI:    {runtime_lib_version[:60]}..."
            + "\n\nThis can cause silent data corruption or crashes. "
            "Please rebuild mpi4jax with the same MPI library you're using at runtime."
        )
        raise RuntimeError(error_msg)


# Run the check on import (can be disabled with environment variable)
if not os.getenv("MPI4JAX_SKIP_ABI_CHECK", "").lower() in ("1", "true", "yes"):
    _check_mpi_abi_compatibility()

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
