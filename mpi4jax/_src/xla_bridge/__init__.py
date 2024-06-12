import os
from jax.lib import xla_client

from . import mpi_xla_bridge
from . import mpi_xla_bridge_cpu

try:
    from . import mpi_xla_bridge_cuda
except ImportError:
    HAS_CUDA_EXT = False
else:
    HAS_CUDA_EXT = True

try:
    from . import mpi_xla_bridge_xpu
except ImportError:
    HAS_XPU_EXT = False
else:
    HAS_XPU_EXT = True

# setup logging


def _is_truthy(str_val):
    return str_val.lower() in ("true", "1", "on")


mpi_xla_bridge.set_logging(_is_truthy(os.getenv("MPI4JAX_DEBUG", "")))


# register custom call targets
for name, fn in mpi_xla_bridge_cpu.custom_call_targets.items():
    xla_client.register_custom_call_target(name, fn, platform="cpu")

if HAS_CUDA_EXT:
    for name, fn in mpi_xla_bridge_cuda.custom_call_targets.items():
        xla_client.register_custom_call_target(name, fn, platform="CUDA")

if HAS_XPU_EXT:
    for name, fn in mpi_xla_bridge_xpu.custom_call_targets.items():
        xla_client.register_custom_call_target(name, fn, platform="SYCL")
