from libc.stdint cimport uintptr_t, int64_t

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


# V2 descriptor builders use int64 for all fields for FFI compatibility
# These match the C++ structs in mpi_descriptors.h

cpdef bytes build_sendrecv_descriptor_v2(
    int64_t sendcount, int64_t dest, int64_t sendtag, int64_t sendtype,
    int64_t recvcount, int64_t source, int64_t recvtag, int64_t recvtype,
    int64_t comm, int64_t status
):
    """Build a sendrecv descriptor using int64 for all fields.

    This descriptor format is used with the FFI API (api_version=4) and is
    compatible with both CPU and CUDA backends.

    All MPI handles (comm, sendtype, recvtype) should be passed as int64
    values obtained from to_mpi_handle().
    """
    cdef SendrecvDescriptorV2 desc = SendrecvDescriptorV2(
        sendcount, dest, sendtag, sendtype,
        recvcount, source, recvtag, recvtype,
        comm, status
    )
    return bytes((<char*> &desc)[:sizeof(SendrecvDescriptorV2)])


def get_sendrecv_descriptor_v2_size():
    """Return the size of SendrecvDescriptorV2 in bytes."""
    return sizeof(SendrecvDescriptorV2)

