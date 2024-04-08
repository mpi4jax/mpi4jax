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


