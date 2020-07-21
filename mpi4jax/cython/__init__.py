from . import mpi_xla_bridge
from jax.lib import xla_client

for name, fn in mpi_xla_bridge.cpu_custom_call_targets.items():
    xla_client.register_cpu_custom_call_target(name, fn)
