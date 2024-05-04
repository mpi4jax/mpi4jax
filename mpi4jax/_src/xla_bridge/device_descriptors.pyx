from libc.stdint cimport uintptr_t

from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
)


cpdef bytes build_recv_descriptor(int nitems, int dest, int tag, uintptr_t comm_handle,
                                  uintptr_t dtype_handle, uintptr_t status_addr):
    cdef RecvDescriptor desc = RecvDescriptor(
        nitems, dest, tag,
        <MPI_Comm> comm_handle,
        <MPI_Datatype> dtype_handle,
        <MPI_Status*> status_addr
    )
    return bytes((<char*> &desc)[:sizeof(RecvDescriptor)])


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


cpdef bytes build_allreduce_descriptor(int nitems, uintptr_t op_handle,
                                       uintptr_t comm_handle, uintptr_t dtype_handle):
    cdef AllreduceDescriptor desc = AllreduceDescriptor(
        nitems, <MPI_Op> op_handle, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(AllreduceDescriptor)])


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


cpdef bytes build_barrier_descriptor(uintptr_t comm_handle):
    cdef BarrierDescriptor desc = BarrierDescriptor(<MPI_Comm> comm_handle)
    return bytes((<char*> &desc)[:sizeof(BarrierDescriptor)])


cpdef bytes build_bcast_descriptor(int nitems, int root, uintptr_t comm_handle,
                                   uintptr_t dtype_handle):
    cdef BcastDescriptor desc = BcastDescriptor(
        nitems, root, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(BcastDescriptor)])


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


cpdef bytes build_reduce_descriptor(int nitems, uintptr_t op_handle, int root,
                                    uintptr_t comm_handle, uintptr_t dtype_handle):
    cdef ReduceDescriptor desc = ReduceDescriptor(
        nitems, <MPI_Op> op_handle, root, <MPI_Comm> comm_handle,
        <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(ReduceDescriptor)])


cpdef bytes build_scan_descriptor(int nitems, uintptr_t op_handle,
                                  uintptr_t comm_handle, uintptr_t dtype_handle):
    cdef ScanDescriptor desc = ScanDescriptor(
        nitems, <MPI_Op> op_handle, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(ScanDescriptor)])


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


cpdef bytes build_send_descriptor(int nitems, int dest, int tag, uintptr_t comm_handle,
                                  uintptr_t dtype_handle):
    cdef SendDescriptor desc = SendDescriptor(
        nitems, dest, tag, <MPI_Comm> comm_handle, <MPI_Datatype> dtype_handle
    )
    return bytes((<char*> &desc)[:sizeof(SendDescriptor)])


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

