# make sure to import mpi4py before anything else
# this calls MPI_Init and registers mpi4py's atexit handler
from mpi4py import MPI  # noqa: F401

# this registers our custom XLA functions
from . import xla_bridge  # noqa: F401

# import public API
from .collective_ops.allgather import allgather  # noqa: F401
from .collective_ops.allreduce import allreduce  # noqa: F401
from .collective_ops.alltoall import alltoall  # noqa: F401
from .collective_ops.barrier import barrier  # noqa: F401
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

JAXLIB_MINIMUM_VERSION = "0.1.62"


def get_version_tuple(verstr):
    # drop everything after the numeric part of the version
    allowed_chars = "0123456789."
    for i, char in enumerate(verstr):
        if char not in allowed_chars:
            break
    else:
        i = len(verstr) + 1

    verstr = verstr[:i].rstrip(".")
    return tuple(int(v) for v in verstr.split("."))[:3]


if get_version_tuple(jaxlib.__version__) < get_version_tuple(JAXLIB_MINIMUM_VERSION):
    raise RuntimeError(
        f"mpi4jax requires jaxlib>={JAXLIB_MINIMUM_VERSION}, but you have "
        f"{jaxlib.__version__}. Please install a supported version of JAX and jaxlib."
    )

del get_version_tuple, jaxlib
