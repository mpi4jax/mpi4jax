# make sure to import mpi4py first
# this calls MPI_Init and registers mpi4py's atexit handler
import mpi4py  # noqa: F401

import mpi4jax.cython  # noqa: F401

from ._create_token import create_token

from .collective_ops.allreduce import Allreduce
from .collective_ops.send import Send
from .collective_ops.recv import Recv
from .collective_ops.sendrecv import Sendrecv

from .warn import disable_omnistaging_warning

# at exit, we wait for all pending operations to finish
# this prevents deadlocks (see mpi4jax#22)
import atexit


@atexit.register
def flush():
    """Wait for all pending XLA operations"""
    import jax

    # as suggested in jax#4335
    noop = jax.device_put(0) + 0
    noop.block_until_ready()


del atexit


__all__ = [
    "Allreduce",
    "Send",
    "Recv",
    "Sendrecv",
    "create_token",
    "disable_omnistaging_warning",
    "flush",
]
