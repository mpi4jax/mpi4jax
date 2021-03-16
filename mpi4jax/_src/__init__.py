# make sure to import mpi4py before anything else
# this calls MPI_Init and registers mpi4py's atexit handler
from mpi4py import MPI  # noqa: F401

# this registers our custom XLA functions
from . import xla_bridge  # noqa: F401

from .collective_ops.allgather import allgather  # noqa: F401
from .collective_ops.allreduce import allreduce  # noqa: F401
from .collective_ops.alltoall import alltoall  # noqa: F401
from .collective_ops.bcast import bcast  # noqa: F401
from .collective_ops.gather import gather  # noqa: F401
from .collective_ops.recv import recv  # noqa: F401
from .collective_ops.reduce import reduce  # noqa: F401
from .collective_ops.scan import scan  # noqa: F401
from .collective_ops.scatter import scatter  # noqa: F401
from .collective_ops.send import send  # noqa: F401
from .collective_ops.sendrecv import sendrecv  # noqa: F401

# check version of jaxlib
import jaxlib
from pkg_resources import Requirement

jaxlib_requirement = Requirement.parse("jaxlib>=0.1.62")

if jaxlib.__version__ not in jaxlib_requirement:
    raise RuntimeError(
        f"mpi4jax requires {jaxlib_requirement}, but you have {jaxlib.__version__}. "
        "Please install a supported version of JAX and jaxlib."
    )
