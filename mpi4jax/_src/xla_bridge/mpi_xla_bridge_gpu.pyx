from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport int32_t, uint64_t
from libc.stdlib cimport malloc, free

from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Comm_rank,
    MPI_Comm_size,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
    MPI_Type_size,
)

from .cuda_runtime_api cimport (
    cudaGetErrorName,
    cudaGetErrorString,
    cudaError_t,
    cudaMemcpy,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyKind,
    cudaMemcpyHostToDevice,
    cudaStream_t,
    cudaStreamSynchronize,
    cudaSuccess,
)

from . cimport mpi_xla_bridge


# Error handling

cpdef inline unicode py_string(char* c_str):
    py_str = <bytes> c_str
    return py_str.decode("UTF-8")


cpdef inline unicode get_error_name(cudaError_t ierr):
    return py_string(cudaGetErrorName(ierr))


cpdef inline unicode get_error_string(cudaError_t ierr):
    return py_string(cudaGetErrorString(ierr))


# Config

cdef bint COPY_TO_HOST = False

cpdef void set_copy_to_host(bint enable):
    global COPY_TO_HOST
    COPY_TO_HOST = enable


# Memory management

cdef inline void* checked_malloc(size_t count) nogil except *:
    cdef void* mem = malloc(count)
    if not mem:
        with gil:
            raise MemoryError(f"Failed to allocate {count} bytes on host")

    return mem


cdef inline cudaError_t checked_cuda_memcpy(void* dst, void* src, size_t count, cudaMemcpyKind kind) nogil except *:
    cdef cudaError_t ierr
    ierr = cudaMemcpy(dst, src, count, kind)

    if ierr != cudaSuccess:
        with gil:
            err_str = get_error_name(ierr)
            err_des = get_error_string(ierr)

            message = f"cudaMemcpy failed with the following error:\n\tError {ierr} {err_str}: {err_des}"
            raise RuntimeError(message)

    return ierr


cdef inline cudaError_t checked_cuda_stream_synchronize(cudaStream_t stream) nogil except *:
    cdef cudaError_t ierr
    ierr = cudaStreamSynchronize(stream)

    if ierr != cudaSuccess:
        with gil:
            err_str = get_error_name(ierr)
            err_des = get_error_string(ierr)

            message = f"cudaMemcpy failed with the following error:\n\tError {ierr} {err_str}: {err_des}"
            raise RuntimeError(message)

    return ierr


#
# GPU XLA targets
#

# Allgather

cdef struct AllgatherDescriptor:
    int32_t sendcount
    MPI_Datatype sendtype
    int32_t recvcount
    MPI_Datatype recvtype
    MPI_Comm comm


cpdef bytes build_allgather_descriptor(
    int32_t sendcount, uint64_t sendtype_addr,
    int32_t recvcount, uint64_t recvtype_addr,
    uint64_t comm_addr
):
    cdef AllgatherDescriptor desc = AllgatherDescriptor(
        sendcount, <MPI_Datatype> sendtype_addr,
        recvcount, <MPI_Datatype> recvtype_addr,
        <MPI_Comm> comm_addr
    )
    return bytes((<char*> &desc)[:sizeof(AllgatherDescriptor)])


cdef void mpi_allgather_gpu(cudaStream_t stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, sendtype_size, recvtype_size, comm_size
    cdef size_t sendbytes, recvbytes

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(AllgatherDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef AllgatherDescriptor* desc = <AllgatherDescriptor*>(opaque)
    cdef int32_t sendcount = desc.sendcount
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int32_t recvcount = desc.recvcount
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef MPI_Comm comm = desc.comm

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(sendtype, &sendtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        sendbytes = sendtype_size * sendcount
        in_buf = checked_malloc(sendbytes)

        ierr = MPI_Type_size(recvtype, &recvtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        ierr = MPI_Comm_size(comm, &comm_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_size")

        # recvcount is received data *per process*
        recvbytes = recvtype_size * recvcount * comm_size
        out_buf = checked_malloc(recvbytes)

        checked_cuda_memcpy(in_buf, data, sendbytes, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_allgather(
        in_buf, sendcount, sendtype,
        out_buf, recvcount, recvtype,
        comm
    )

    if COPY_TO_HOST:
        # copy back to device
        checked_cuda_memcpy(out_data, out_buf, recvbytes, cudaMemcpyHostToDevice)
        free(in_buf)
        free(out_buf)


# Allreduce

cdef struct AllreduceDescriptor:
    int32_t nitems
    MPI_Op op
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_allreduce_descriptor(int32_t nitems, uint64_t op_addr, uint64_t comm_addr, uint64_t dtype_addr):
    cdef AllreduceDescriptor desc = AllreduceDescriptor(
        nitems, <MPI_Op> op_addr, <MPI_Comm> comm_addr, <MPI_Datatype> dtype_addr
    )
    return bytes((<char*> &desc)[:sizeof(AllreduceDescriptor)])


cdef void mpi_allreduce_gpu(cudaStream_t stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size
    cdef size_t count

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(AllreduceDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef AllreduceDescriptor* desc = <AllreduceDescriptor*>(opaque)
    cdef int32_t nitems = desc.nitems
    cdef MPI_Op op = desc.op
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        in_buf = checked_malloc(count)
        out_buf = checked_malloc(count)
        checked_cuda_memcpy(in_buf, data, count, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_allreduce(in_buf, out_buf, nitems, dtype, op, comm)

    if COPY_TO_HOST:
        # copy back to device
        checked_cuda_memcpy(out_data, out_buf, count, cudaMemcpyHostToDevice)
        free(in_buf)
        free(out_buf)


# Alltoall

cdef struct AlltoallDescriptor:
    int32_t sendcount
    MPI_Datatype sendtype
    int32_t recvcount
    MPI_Datatype recvtype
    MPI_Comm comm


cpdef bytes build_alltoall_descriptor(
    int32_t sendcount, uint64_t sendtype_addr,
    int32_t recvcount, uint64_t recvtype_addr,
    uint64_t comm_addr
):
    cdef AlltoallDescriptor desc = AlltoallDescriptor(
        sendcount, <MPI_Datatype> sendtype_addr,
        recvcount, <MPI_Datatype> recvtype_addr,
        <MPI_Comm> comm_addr
    )
    return bytes((<char*> &desc)[:sizeof(AlltoallDescriptor)])


cdef void mpi_alltoall_gpu(cudaStream_t stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, sendtype_size, recvtype_size, comm_size
    cdef size_t sendbytes, recvbytes

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(AlltoallDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef AlltoallDescriptor* desc = <AlltoallDescriptor*>(opaque)
    cdef int32_t sendcount = desc.sendcount
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int32_t recvcount = desc.recvcount
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef MPI_Comm comm = desc.comm

    cdef float first_val

    if COPY_TO_HOST:
        # copy memory to host

        # counts are *per process*, so multiply with comm_size
        ierr = MPI_Comm_size(comm, &comm_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_size")

        ierr = MPI_Type_size(sendtype, &sendtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        sendbytes = sendtype_size * sendcount * comm_size
        in_buf = checked_malloc(sendbytes)

        ierr = MPI_Type_size(recvtype, &recvtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        recvbytes = recvtype_size * recvcount * comm_size
        out_buf = checked_malloc(recvbytes)

        checked_cuda_memcpy(in_buf, data, sendbytes, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_alltoall(
        in_buf, sendcount, sendtype, out_buf, recvcount, recvtype, comm
    )

    if COPY_TO_HOST:
        # copy back to device
        checked_cuda_memcpy(out_data, out_buf, recvbytes, cudaMemcpyHostToDevice)
        free(in_buf)
        free(out_buf)


# Bcast

cdef struct BcastDescriptor:
    int32_t nitems
    int32_t root
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_bcast_descriptor(int32_t nitems, int32_t root, uint64_t comm_addr, uint64_t dtype_addr):
    cdef BcastDescriptor desc = BcastDescriptor(
        nitems, root, <MPI_Comm> comm_addr, <MPI_Datatype> dtype_addr
    )
    return bytes((<char*> &desc)[:sizeof(BcastDescriptor)])


cdef void mpi_bcast_gpu(cudaStream_t stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size, rank
    cdef size_t count

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* buf = out_data

    if opaque_len != sizeof(BcastDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef BcastDescriptor* desc = <BcastDescriptor*>(opaque)
    cdef int32_t nitems = desc.nitems
    cdef int32_t root = desc.root
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    ierr = MPI_Type_size(dtype, &dtype_size)
    mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")
    ierr = MPI_Comm_rank(comm, &rank)
    mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_rank")

    if COPY_TO_HOST:
        # copy memory to host
        count = dtype_size * nitems
        buf = checked_malloc(count)
        if rank == root:
            checked_cuda_memcpy(buf, data, count, cudaMemcpyDeviceToHost)
    else:
        if rank == root:
            buf = data

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_bcast(buf, nitems, dtype, root, comm)

    if COPY_TO_HOST:
        if rank != root:
            # copy back to device
            checked_cuda_memcpy(out_data, buf, count, cudaMemcpyHostToDevice)
        free(buf)


# Gather

cdef struct GatherDescriptor:
    int32_t sendcount
    MPI_Datatype sendtype
    int32_t recvcount
    MPI_Datatype recvtype
    int32_t root
    MPI_Comm comm


cpdef bytes build_gather_descriptor(
    int32_t sendcount, uint64_t sendtype_addr,
    int32_t recvcount, uint64_t recvtype_addr,
    int32_t root, uint64_t comm_addr
):
    cdef GatherDescriptor desc = GatherDescriptor(
        sendcount, <MPI_Datatype> sendtype_addr,
        recvcount, <MPI_Datatype> recvtype_addr,
        root, <MPI_Comm> comm_addr
    )
    return bytes((<char*> &desc)[:sizeof(GatherDescriptor)])


cdef void mpi_gather_gpu(cudaStream_t stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, sendtype_size, recvtype_size, rank, size
    cdef size_t sendbytes, recvbytes

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(GatherDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef GatherDescriptor* desc = <GatherDescriptor*>(opaque)
    cdef int32_t sendcount = desc.sendcount
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int32_t recvcount = desc.recvcount
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef int32_t root = desc.root
    cdef MPI_Comm comm = desc.comm

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(sendtype, &sendtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        sendbytes = sendtype_size * sendcount
        in_buf = checked_malloc(sendbytes)

        ierr = MPI_Type_size(recvtype, &recvtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        ierr = MPI_Comm_rank(comm, &rank)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_rank")

        ierr = MPI_Comm_size(comm, &size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_size")

        recvbytes = recvtype_size * recvcount
        if rank == root:
            # output shape is larger on root
            recvbytes *= size

        out_buf = checked_malloc(recvbytes)

        checked_cuda_memcpy(in_buf, data, sendbytes, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_gather(
        in_buf, sendcount, sendtype, out_buf, recvcount, recvtype, root, comm
    )

    if COPY_TO_HOST:
        if rank == root:
            # copy back to device
            checked_cuda_memcpy(out_data, out_buf, recvbytes, cudaMemcpyHostToDevice)

        free(in_buf)
        free(out_buf)


# Recv

cdef struct RecvDescriptor:
    int32_t nitems
    int32_t source
    int32_t tag
    MPI_Comm comm
    MPI_Datatype dtype
    MPI_Status* status


cpdef bytes build_recv_descriptor(int32_t nitems, int32_t dest, int32_t tag, uint64_t comm_addr,
                                  uint64_t dtype_addr, uint64_t status_addr):
    cdef RecvDescriptor desc = RecvDescriptor(
        nitems, dest, tag, <MPI_Comm> comm_addr, <MPI_Datatype> dtype_addr, <MPI_Status*> status_addr
    )
    return bytes((<char*> &desc)[:sizeof(RecvDescriptor)])


cdef void mpi_recv_gpu(cudaStream_t stream, void** buffers,
                   const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size
    cdef size_t count

    #decode inputs
    cdef void* out_buf = buffers[1]

    cdef void* recvbuf = out_buf

    if opaque_len != sizeof(RecvDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef RecvDescriptor* desc = <RecvDescriptor*>(opaque)
    cdef int32_t nitems = desc.nitems
    cdef int32_t source = desc.source
    cdef int32_t tag = desc.tag
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype
    cdef MPI_Status* status = desc.status

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        recvbuf = checked_malloc(count)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_recv(recvbuf, nitems, dtype, source, tag, comm, status)

    if COPY_TO_HOST:
        checked_cuda_memcpy(out_buf, recvbuf, count, cudaMemcpyHostToDevice)
        free(recvbuf)


# Reduce

cdef struct ReduceDescriptor:
    int32_t nitems
    MPI_Op op
    int32_t root
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_reduce_descriptor(int32_t nitems, uint64_t op_addr, int32_t root, uint64_t comm_addr, uint64_t dtype_addr):
    cdef ReduceDescriptor desc = ReduceDescriptor(
        nitems, <MPI_Op> op_addr, root, <MPI_Comm> comm_addr, <MPI_Datatype> dtype_addr
    )
    return bytes((<char*> &desc)[:sizeof(ReduceDescriptor)])


cdef void mpi_reduce_gpu(cudaStream_t stream, void** buffers,
                     const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size, rank
    cdef size_t count

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(ReduceDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef ReduceDescriptor* desc = <ReduceDescriptor*>(opaque)
    cdef int32_t nitems = desc.nitems
    cdef MPI_Op op = desc.op
    cdef int32_t root = desc.root
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        in_buf = checked_malloc(count)
        out_buf = checked_malloc(count)
        checked_cuda_memcpy(in_buf, data, count, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_reduce(in_buf, out_buf, nitems, dtype, op, root, comm)

    if COPY_TO_HOST:
        ierr = MPI_Comm_rank(comm, &rank)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_rank")

        if rank == root:
            # copy back to device
            checked_cuda_memcpy(out_data, out_buf, count, cudaMemcpyHostToDevice)

        free(in_buf)
        free(out_buf)


# Scan

cdef struct ScanDescriptor:
    int32_t nitems
    MPI_Op op
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_scan_descriptor(int32_t nitems, uint64_t op_addr, uint64_t comm_addr, uint64_t dtype_addr):
    cdef ScanDescriptor desc = ScanDescriptor(
        nitems, <MPI_Op> op_addr, <MPI_Comm> comm_addr, <MPI_Datatype> dtype_addr
    )
    return bytes((<char*> &desc)[:sizeof(ScanDescriptor)])


cdef void mpi_scan_gpu(cudaStream_t stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size
    cdef size_t count

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(ScanDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef ScanDescriptor* desc = <ScanDescriptor*>(opaque)
    cdef int32_t nitems = desc.nitems
    cdef MPI_Op op = desc.op
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        in_buf = checked_malloc(count)
        out_buf = checked_malloc(count)
        checked_cuda_memcpy(in_buf, data, count, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_scan(in_buf, out_buf, nitems, dtype, op, comm)

    if COPY_TO_HOST:
        # copy back to device
        checked_cuda_memcpy(out_data, out_buf, count, cudaMemcpyHostToDevice)
        free(in_buf)
        free(out_buf)


# Scatter

cdef struct ScatterDescriptor:
    int32_t sendcount
    MPI_Datatype sendtype
    int32_t recvcount
    MPI_Datatype recvtype
    int32_t root
    MPI_Comm comm


cpdef bytes build_scatter_descriptor(
    int32_t sendcount, uint64_t sendtype_addr,
    int32_t recvcount, uint64_t recvtype_addr,
    int32_t root, uint64_t comm_addr
):
    cdef ScatterDescriptor desc = ScatterDescriptor(
        sendcount, <MPI_Datatype> sendtype_addr,
        recvcount, <MPI_Datatype> recvtype_addr,
        root, <MPI_Comm> comm_addr
    )
    return bytes((<char*> &desc)[:sizeof(ScatterDescriptor)])


cdef void mpi_scatter_gpu(cudaStream_t stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, sendtype_size, recvtype_size, rank, size
    cdef size_t sendbytes, recvbytes

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(ScatterDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef ScatterDescriptor* desc = <ScatterDescriptor*>(opaque)
    cdef int32_t sendcount = desc.sendcount
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int32_t recvcount = desc.recvcount
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef int32_t root = desc.root
    cdef MPI_Comm comm = desc.comm

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Comm_rank(comm, &rank)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_rank")

        ierr = MPI_Comm_size(comm, &size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_size")

        ierr = MPI_Type_size(sendtype, &sendtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        sendbytes = sendtype_size * sendcount
        if rank == root:
            # input is larger on root
            sendbytes *= size

        in_buf = checked_malloc(sendbytes)

        ierr = MPI_Type_size(recvtype, &recvtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        recvbytes = recvtype_size * recvcount
        out_buf = checked_malloc(recvbytes)

        checked_cuda_memcpy(in_buf, data, sendbytes, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_scatter(
        in_buf, sendcount, sendtype, out_buf, recvcount, recvtype, root, comm
    )

    if COPY_TO_HOST:
        # copy back to device
        checked_cuda_memcpy(out_data, out_buf, recvbytes, cudaMemcpyHostToDevice)

        free(in_buf)
        free(out_buf)


# Send

cdef struct SendDescriptor:
    int32_t nitems
    int32_t dest
    int32_t tag
    MPI_Comm comm
    MPI_Datatype dtype


cpdef bytes build_send_descriptor(int32_t nitems, int32_t dest, int32_t tag, uint64_t comm_addr, uint64_t dtype_addr):
    cdef SendDescriptor desc = SendDescriptor(
        nitems, dest, tag, <MPI_Comm> comm_addr, <MPI_Datatype> dtype_addr
    )
    return bytes((<char*> &desc)[:sizeof(SendDescriptor)])


cdef void mpi_send_gpu(cudaStream_t stream, void** buffers,
                   const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size
    cdef size_t count

    #decode inputs
    cdef void* data = buffers[0]

    cdef void* sendbuf = data

    if opaque_len != sizeof(SendDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef SendDescriptor* desc = <SendDescriptor*>(opaque)
    cdef int32_t nitems = desc.nitems
    cdef int32_t dest = desc.dest
    cdef int32_t tag = desc.tag
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        sendbuf = checked_malloc(count)
        checked_cuda_memcpy(sendbuf, data, count, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_send(sendbuf, nitems, dtype, dest, tag, comm)

    if COPY_TO_HOST:
        free(sendbuf)


# Sendrecv

cdef struct SendrecvDescriptor:
    int32_t sendcount
    int32_t dest
    int32_t sendtag
    MPI_Datatype sendtype
    int32_t recvcount
    int32_t source
    int32_t recvtag
    MPI_Datatype recvtype
    MPI_Comm comm
    MPI_Status* status


cpdef bytes build_sendrecv_descriptor(int32_t sendcount, int32_t dest, int32_t sendtag, uint64_t sendtype_addr,
                                      int32_t recvcount, int32_t source, int32_t recvtag, uint64_t recvtype_addr,
                                      uint64_t comm_addr, uint64_t status_addr):
    cdef SendrecvDescriptor desc = SendrecvDescriptor(
        sendcount, dest, sendtag, <MPI_Datatype> sendtype_addr,
        recvcount, source, recvtag, <MPI_Datatype> recvtype_addr,
        <MPI_Comm> comm_addr, <MPI_Status*> status_addr
    )
    return bytes((<char*> &desc)[:sizeof(SendrecvDescriptor)])


cdef void mpi_sendrecv_gpu(cudaStream_t stream, void** buffers,
                       const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, send_dtype_size, recv_dtype_size
    cdef size_t bytes_send, bytes_recv

    #decode inputs
    cdef void* in_buf = buffers[0]
    cdef void* out_buf = buffers[2]

    cdef void* sendbuf = in_buf
    cdef void* recvbuf = out_buf

    if opaque_len != sizeof(SendrecvDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef SendrecvDescriptor* desc = <SendrecvDescriptor*>(opaque)
    cdef int32_t sendcount = desc.sendcount
    cdef int32_t dest = desc.dest
    cdef int32_t sendtag = desc.sendtag
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int32_t recvcount = desc.recvcount
    cdef int32_t source = desc.source
    cdef int32_t recvtag = desc.recvtag
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Status* status = desc.status

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(sendtype, &send_dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        ierr = MPI_Type_size(recvtype, &recv_dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        bytes_send = send_dtype_size * sendcount
        bytes_recv = recv_dtype_size * recvcount
        sendbuf = checked_malloc(bytes_send)
        recvbuf = checked_malloc(bytes_recv)
        checked_cuda_memcpy(sendbuf, in_buf, bytes_send, cudaMemcpyDeviceToHost)

    checked_cuda_stream_synchronize(stream)
    mpi_xla_bridge.mpi_sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    )

    if COPY_TO_HOST:
        checked_cuda_memcpy(out_buf, recvbuf, bytes_recv, cudaMemcpyHostToDevice)
        free(recvbuf)


gpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    gpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)


register_custom_call_target(b"mpi_allgather", <void*>(mpi_allgather_gpu))
register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce_gpu))
register_custom_call_target(b"mpi_alltoall", <void*>(mpi_alltoall_gpu))
register_custom_call_target(b"mpi_bcast", <void*>(mpi_bcast_gpu))
register_custom_call_target(b"mpi_gather", <void*>(mpi_gather_gpu))
register_custom_call_target(b"mpi_recv", <void*>(mpi_recv_gpu))
register_custom_call_target(b"mpi_reduce", <void*>(mpi_reduce_gpu))
register_custom_call_target(b"mpi_scan", <void*>(mpi_scan_gpu))
register_custom_call_target(b"mpi_scatter", <void*>(mpi_scatter_gpu))
register_custom_call_target(b"mpi_send", <void*>(mpi_send_gpu))
register_custom_call_target(b"mpi_sendrecv", <void*>(mpi_sendrecv_gpu))
