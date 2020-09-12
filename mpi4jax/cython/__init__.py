from . import mpi_xla_bridge
from jax.lib import xla_client

import os

mpi_xla_bridge.set_logging(
    os.environ.get("MPI4JAX_DEBUG", "").lower() in ("true", "1", "on")
)

for name, fn in mpi_xla_bridge.cpu_custom_call_targets.items():
    xla_client.register_cpu_custom_call_target(name, fn)

del os, xla_client
