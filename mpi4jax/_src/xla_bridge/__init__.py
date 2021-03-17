from jax.lib import xla_client

from . import mpi_xla_bridge_cpu

try:
    from . import mpi_xla_bridge_gpu
except ImportError:
    HAS_GPU_EXT = False
else:
    HAS_GPU_EXT = True

# register custom call targets
for name, fn in mpi_xla_bridge_cpu.cpu_custom_call_targets.items():
    xla_client.register_custom_call_target(name, fn, platform="cpu")

if HAS_GPU_EXT:
    for name, fn in mpi_xla_bridge_gpu.gpu_custom_call_targets.items():
        xla_client.register_custom_call_target(name, fn, platform="gpu")
