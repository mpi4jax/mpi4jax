from libc.stdint cimport uintptr_t

from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Comm_rank,
    MPI_Comm_size,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
    MPI_Type_size,
)

# Recv

cdef struct RecvDescriptor:
    int nitems
    int source
    int tag
    MPI_Comm comm
    MPI_Datatype dtype
    MPI_Status* status


cpdef bytes build_recv_descriptor(int nitems, int dest, int tag, uintptr_t comm_handle,
                                  uintptr_t dtype_handle, uintptr_t status_addr):
    cdef RecvDescriptor desc = RecvDescriptor(
        nitems, dest, tag,
        <MPI_Comm> comm_handle,
        <MPI_Datatype> dtype_handle,
        <MPI_Status*> status_addr
    )
    return bytes((<char*> &desc)[:sizeof(RecvDescriptor)])


# Allgather

cdef struct AllgatherDescriptor:
    int sendcount
    MPI_Datatype sendtype
    int recvcount
    MPI_Datatype recvtype
    MPI_Comm comm


cpdef bytes build_allgather_descriptor(
    int sendcount, uintptr_t sendtype_handle,
    int recvcount, uintptr_t recvtype_addr,
    uintptr_t comm_handle
):
    cdef AllgatherDescriptor desc = AllgatherDescriptor(
        sendcount, <MPI_Datatype> sendtype_handle,
        recvcount, <MPI_Datatype> recvtype_addr,
        <MPI_Comm> comm_handle
    )
    return bytes((<char*> &desc)[:sizeof(AllgatherDescriptor)])

# Allreduce

cdef struct AllreduceDescriptor:
    int nitems
    MPI_Op op
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_allreduce_descriptor(int nitems, uintptr_t op_handle,
                                       uintptr_t comm_handle, uintptr_t dtype_handle):
    cdef AllreduceDescriptor desc = AllreduceDescriptor(
        nitems, <MPI_Op> op_handle, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(AllreduceDescriptor)])

# Alltoall

cdef struct AlltoallDescriptor:
    int sendcount
    MPI_Datatype sendtype
    int recvcount
    MPI_Datatype recvtype
    MPI_Comm comm


cpdef bytes build_alltoall_descriptor(
    int sendcount, uintptr_t sendtype_handle,
    int recvcount, uintptr_t recvtype_addr,
    uintptr_t comm_handle
):
    cdef AlltoallDescriptor desc = AlltoallDescriptor(
        sendcount, <MPI_Datatype> sendtype_handle,
        recvcount, <MPI_Datatype> recvtype_addr,
        <MPI_Comm> comm_handle
    )
    return bytes((<char*> &desc)[:sizeof(AlltoallDescriptor)])

# Barrier

cdef struct BarrierDescriptor:
    MPI_Comm comm


cpdef bytes build_barrier_descriptor(uintptr_t comm_handle):
    cdef BarrierDescriptor desc = BarrierDescriptor(<MPI_Comm> comm_handle)
    return bytes((<char*> &desc)[:sizeof(BarrierDescriptor)])

# Bcast

cdef struct BcastDescriptor:
    int nitems
    int root
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_bcast_descriptor(int nitems, int root, uintptr_t comm_handle,
                                   uintptr_t dtype_handle):
    cdef BcastDescriptor desc = BcastDescriptor(
        nitems, root, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(BcastDescriptor)])

# Gather

cdef struct GatherDescriptor:
    int sendcount
    MPI_Datatype sendtype
    int recvcount
    MPI_Datatype recvtype
    int root
    MPI_Comm comm


cpdef bytes build_gather_descriptor(
    int sendcount, uintptr_t sendtype_handle,
    int recvcount, uintptr_t recvtype_addr,
    int root, uintptr_t comm_handle
):
    cdef GatherDescriptor desc = GatherDescriptor(
        sendcount, <MPI_Datatype> sendtype_handle,
        recvcount, <MPI_Datatype> recvtype_addr,
        root, <MPI_Comm> comm_handle
    )
    return bytes((<char*> &desc)[:sizeof(GatherDescriptor)])

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


cpdef bytes build_scan_descriptor(int nitems, uintptr_t op_handle,
                                  uintptr_t comm_handle, uintptr_t dtype_handle):
    cdef ScanDescriptor desc = ScanDescriptor(
        nitems, <MPI_Op> op_handle, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(ScanDescriptor)])


# Scatter

cdef struct ScatterDescriptor:
    int sendcount
    MPI_Datatype sendtype
    int recvcount
    MPI_Datatype recvtype
    int root
    MPI_Comm comm


cpdef bytes build_scatter_descriptor(
    int sendcount, uintptr_t sendtype_handle,
    int recvcount, uintptr_t recvtype_addr,
    int root, uintptr_t comm_handle
):
    cdef ScatterDescriptor desc = ScatterDescriptor(
        sendcount, <MPI_Datatype> sendtype_handle,
        recvcount, <MPI_Datatype> recvtype_addr,
        root, <MPI_Comm> comm_handle
    )
    return bytes((<char*> &desc)[:sizeof(ScatterDescriptor)])


# Send

cdef struct SendDescriptor:
    int nitems
    int dest
    int tag
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_send_descriptor(int nitems, int dest, int tag, uintptr_t comm_handle,
                                  uintptr_t dtype_handle):
    cdef SendDescriptor desc = SendDescriptor(
        nitems, dest, tag, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(SendDescriptor)])


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


cpdef bytes build_sendrecv_descriptor(
    int sendcount, int dest, int sendtag, uintptr_t sendtype_handle,
    int recvcount, int source, int recvtag, uintptr_t recvtype_addr,
    uintptr_t comm_handle, uintptr_t status_addr
):
    cdef SendrecvDescriptor desc = SendrecvDescriptor(
        sendcount, dest, sendtag, <MPI_Datatype> sendtype_handle,
        recvcount, source, recvtag, <MPI_Datatype> recvtype_addr,
        <MPI_Comm> comm_handle, <MPI_Status*> status_addr
    )
    return bytes((<char*> &desc)[:sizeof(SendrecvDescriptor)])

