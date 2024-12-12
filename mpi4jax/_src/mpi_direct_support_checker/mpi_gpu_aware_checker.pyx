# mpi_check.pyx
cdef extern from "mpi_gpu_aware.h":
    int mpi_gpu_aware()

def is_gpu_aware_mpi():
    return mpi_gpu_aware()