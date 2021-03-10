# make sure to import mpi4py first
# this calls MPI_Init and registers mpi4py's atexit handler
from mpi4py import MPI  # noqa: F401

# this registers our custom XLA functions
from . import cython  # noqa: F401

from .collective_ops import allreduce, bcast, send, recv, sendrecv
from .flush import flush
from .warn import disable_omnistaging_warning

# at exit, we wait for all pending operations to finish
# this prevents deadlocks (see mpi4jax#22)
import atexit

atexit.register(flush)


__all__ = [
    "allreduce",
    "send",
    "recv",
    "sendrecv",
    "bcast",
    "disable_omnistaging_warning",
    "flush",
]

del MPI
