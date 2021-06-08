# this defines the public API of mpi4jax
# all other intialization is taking place in _src/__init__.py

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from ._src import (  # noqa: E402
    allgather,
    allreduce,
    alltoall,
    barrier,
    bcast,
    gather,
    recv,
    reduce,
    scan,
    scatter,
    send,
    sendrecv,
)

__all__ = [
    "allgather",
    "allreduce",
    "alltoall",
    "barrier",
    "bcast",
    "gather",
    "recv",
    "reduce",
    "scan",
    "scatter",
    "send",
    "sendrecv",
]
