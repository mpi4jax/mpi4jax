# make sure to import mpi4py first
# this calls MPI_Init and registers mpi4py's atexit handler
from mpi4py import MPI

# this registers our custom XLA functions
import mpi4jax.cython

from ._create_token import create_token

from .collective_ops.allreduce import Allreduce
from .collective_ops.send import Send
from .collective_ops.recv import Recv
from .collective_ops.sendrecv import Sendrecv

from .warn import disable_omnistaging_warning
from .flush import flush

# at exit, we wait for all pending operations to finish
# this prevents deadlocks (see mpi4jax#22)
import atexit

atexit.register(flush)


__all__ = [
    "Allreduce",
    "Send",
    "Recv",
    "Sendrecv",
    "create_token",
    "disable_omnistaging_warning",
    "flush",
]

del atexit, MPI, mpi4jax.cython
