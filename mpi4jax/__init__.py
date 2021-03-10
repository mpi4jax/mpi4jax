from ._version import version as __version__  # noqa: F401

from ._src import allreduce, recv, send, sendrecv, bcast, disable_omnistaging_warning

from ._deprecations import Allreduce, Recv, Send, Sendrecv, Bcast

__all__ = [
    "allreduce",
    "send",
    "recv",
    "sendrecv",
    "bcast",
    "disable_omnistaging_warning",
    "Allreduce",
    "Recv",
    "Send",
    "Sendrecv",
    "Bcast",
]
