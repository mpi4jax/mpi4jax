# this defines the public API of mpi4jax
# all other intialization is taking place in _src/__init__.py

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from ._src import (  # noqa: E402
    allgather,
    allreduce,
    alltoall,
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
    "bcast",
    "gather",
    "recv",
    "reduce",
    "scan",
    "scatter",
    "send",
    "sendrecv",
]

# TODO: remove in next minor release
from ._deprecations import (  # noqa: E402
    Allreduce,
    Bcast,
    Recv,
    Send,
    Sendrecv,
)

__all__.extend(
    [
        Allreduce,
        Bcast,
        Recv,
        Send,
        Sendrecv,
    ]
)
