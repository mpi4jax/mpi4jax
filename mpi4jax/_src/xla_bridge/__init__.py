import os

import jax

from . import mpi_xla_bridge
from . import mpi_xla_bridge_cpu

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

# TODO: move to jax.extend.ffi interface once we deprecate
# jax 0.4.34
if jax.__version_info__ >= (0, 4, 35):
    import jax.extend as jex

    def register_custom_call_target(name, fn, *, platform: str, api_version: int):
        return jex.ffi.register_ffi_target(
            name, fn, platform=platform, api_version=api_version
        )

else:
    from jax.lib import xla_client

    def register_custom_call_target(name, fn, *, platform: str, api_version: int):
        if api_version != 0:
            raise NotImplementedError("only api version 0 supported")
        return xla_client.register_custom_call_target(name, fn, platform=platform)


def _is_truthy(str_val):
    return str_val.lower() in ("true", "1", "on")


mpi_xla_bridge.set_logging(_is_truthy(os.getenv("MPI4JAX_DEBUG", "")))


# register custom call targets
for name, fn in mpi_xla_bridge_cpu.custom_call_targets.items():
    register_custom_call_target(name, fn, platform="cpu", api_version=0)

if HAS_CUDA_EXT:
    for name, fn in mpi_xla_bridge_cuda.custom_call_targets.items():
        register_custom_call_target(name, fn, platform="CUDA", api_version=0)

# if HAS_XPU_EXT:
#    for name, fn in mpi_xla_bridge_xpu.custom_call_targets.items():
#        register_custom_call_target(name, fn, platform="SYCL", api_version=0)
