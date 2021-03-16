import atexit
import threading

_default_comm = threading.local()
_default_comm.value = None


def get_default_comm():
    if _default_comm.value is None:
        from mpi4py import MPI

        default_comm = MPI.COMM_WORLD.Clone()
        atexit.register(default_comm.Free)
        _default_comm.value = default_comm

    return _default_comm.value
