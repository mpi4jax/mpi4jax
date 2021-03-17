def pytest_report_header(config):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return [
        f"MPI vendor: {MPI.get_vendor()}",
        f"MPI rank: {comm.Get_rank()}",
        f"MPI size: {comm.Get_size()}",
    ]


def pytest_configure(config):
    # don't hog memory if running on GPU
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
