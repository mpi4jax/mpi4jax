from ._version import version as __version__  # noqa: F401

# make sure to import mpi4py first
# this calls MPI_Init and registers mpi4py's atexit handler
from mpi4py import MPI

# this registers our custom XLA functions
import mpi4jax.cython

from .collective_ops.allgather import Allgather
from .collective_ops.allreduce import Allreduce
from .collective_ops.alltoall import Alltoall
from .collective_ops.bcast import Bcast
from .collective_ops.gather import Gather
from .collective_ops.recv import Recv
from .collective_ops.reduce import Reduce
from .collective_ops.scan import Scan
from .collective_ops.scatter import Scatter
from .collective_ops.send import Send
from .collective_ops.sendrecv import Sendrecv

from .flush import flush
from .warn import disable_omnistaging_warning

# at exit, we wait for all pending operations to finish
# this prevents deadlocks (see mpi4jax#22)
import atexit

atexit.register(flush)


__all__ = [
    "Allgather",
    "Allreduce",
    "Alltoall",
    "Bcast",
    "Gather",
    "Recv",
    "Reduce",
    "Scan",
    "Scatter",
    "Send",
    "Sendrecv",
    "disable_omnistaging_warning",
    "flush",
]

del atexit, MPI, mpi4jax.cython
