from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
)

# Recv

cdef struct RecvDescriptor:
    int nitems
    int source
    int tag
    MPI_Comm comm
    MPI_Datatype dtype
    MPI_Status* status


# Allgather

cdef struct AllgatherDescriptor:
    int sendcount
    MPI_Datatype sendtype
    int recvcount
    MPI_Datatype recvtype
    MPI_Comm comm


# Allreduce

cdef struct AllreduceDescriptor:
    int nitems
    MPI_Op op
    MPI_Comm comm
    MPI_Datatype dtype


# Alltoall

cdef struct AlltoallDescriptor:
    int sendcount
    MPI_Datatype sendtype
    int recvcount
    MPI_Datatype recvtype
    MPI_Comm comm


# Barrier

cdef struct BarrierDescriptor:
    MPI_Comm comm


# Bcast

cdef struct BcastDescriptor:
    int nitems
    int root
    MPI_Comm comm
    MPI_Datatype dtype


# Gather

cdef struct GatherDescriptor:
    int sendcount
    MPI_Datatype sendtype
    int recvcount
    MPI_Datatype recvtype
    int root
    MPI_Comm comm


# Reduce

cdef struct ReduceDescriptor:
    int nitems
    MPI_Op op
    int root
    MPI_Comm comm
    MPI_Datatype dtype


# Scan

cdef struct ScanDescriptor:
    int nitems
    MPI_Op op
    MPI_Comm comm
    MPI_Datatype dtype


# Scatter

cdef struct ScatterDescriptor:
    int sendcount
    MPI_Datatype sendtype
    int recvcount
    MPI_Datatype recvtype
    int root
    MPI_Comm comm


# Send

cdef struct SendDescriptor:
    int nitems
    int dest
    int tag
    MPI_Comm comm
    MPI_Datatype dtype


# Sendrecv

cdef struct SendrecvDescriptor:
    int sendcount
    int dest
    int sendtag
    MPI_Datatype sendtype
    int recvcount
    int source
    int recvtag
    MPI_Datatype recvtype
    MPI_Comm comm
    MPI_Status* status


# SendrecvDescriptorV2 - uses int64 for all fields for FFI compatibility
# This matches the C++ mpi_descriptors.h SendrecvDescriptor struct
from libc.stdint cimport int64_t

cdef packed struct SendrecvDescriptorV2:
    int64_t sendcount
    int64_t dest
    int64_t sendtag
    int64_t sendtype     # MPI_Datatype handle as int64
    int64_t recvcount
    int64_t source
    int64_t recvtag
    int64_t recvtype     # MPI_Datatype handle as int64
    int64_t comm         # MPI_Comm handle as int64
    int64_t status       # MPI_Status* pointer as int64


