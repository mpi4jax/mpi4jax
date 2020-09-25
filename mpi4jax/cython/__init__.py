import os
import warnings

from jax.lib import xla_client

from . import mpi_xla_bridge
from . import mpi_xla_bridge_cpu

try:
    from . import mpi_xla_bridge_gpu
except ImportError:
    HAS_GPU_EXT = False
else:
    HAS_GPU_EXT = True


def is_truthy(str_val):
    return str_val.lower() in ("true", "1", "on")


def is_falsy(str_val):
    return str_val.lower() in ("false", "0", "off")


enable_logging = is_truthy(os.environ.get("MPI4JAX_DEBUG", ""))
mpi_xla_bridge.set_logging(enable_logging)


if HAS_GPU_EXT:
    gpu_copy_behavior = os.environ.get("MPI4JAX_USE_CUDA_MPI", "")

    if is_truthy(gpu_copy_behavior):
        has_cuda_mpi = True
    elif is_falsy(gpu_copy_behavior):
        has_cuda_mpi = False
    else:
        has_cuda_mpi = False
        warn_msg = (
            "Not using CUDA-enabled MPI. "
            "If you are sure that your MPI library is built with CUDA support, "
            "set MPI4JAX_USE_CUDA_MPI=1. To silence this warning, "
            "set MPI4JAX_USE_CUDA_MPI=0."
        )
        warnings.warn(warn_msg)

    mpi_xla_bridge_gpu.set_copy_to_host(not has_cuda_mpi)


# register custom call targets
for name, fn in mpi_xla_bridge_cpu.cpu_custom_call_targets.items():
    xla_client.register_custom_call_target(name, fn, platform="cpu")

if HAS_GPU_EXT:
    for name, fn in mpi_xla_bridge_gpu.gpu_custom_call_targets.items():
        xla_client.register_custom_call_target(name, fn, platform="gpu")


del os, xla_client
