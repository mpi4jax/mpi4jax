_default_comm = None


def get_default_comm():
    global _default_comm
    if _default_comm is None:
        from mpi4py import MPI

        _default_comm = MPI.COMM_WORLD.Clone()

    return _default_comm
