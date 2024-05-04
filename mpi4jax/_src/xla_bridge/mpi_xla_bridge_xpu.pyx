from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free

from cython.operator cimport dereference as deref

from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Comm_rank,
    MPI_Comm_size,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
    MPI_Type_size,
)

from .device_descriptors cimport (
    RecvDescriptor,
    AllgatherDescriptor,
    AllreduceDescriptor,
    AlltoallDescriptor,
    BarrierDescriptor,
    BcastDescriptor,
    GatherDescriptor,
    ReduceDescriptor,
    ScanDescriptor,
    ScatterDescriptor,
    SendDescriptor,
    SendrecvDescriptor,
)

from .sycl_runtime_api cimport (
    queue,
    event,
)

from . cimport mpi_xla_bridge


# Config

cdef bint COPY_TO_HOST = False

cpdef void set_copy_to_host(bint enable):
    global COPY_TO_HOST
    COPY_TO_HOST = enable


# Memory management

cdef inline void* checked_malloc(size_t count, MPI_Comm comm) nogil:
    cdef void* mem = malloc(count)
    if not mem:
        with gil:
            message = f"Failed to allocate {count} bytes on host"

        mpi_xla_bridge.abort(0, comm, message)

    return mem


cdef inline checked_sycl_memcpy(queue* sycl_queue, void* dst, void* src, size_t count, MPI_Comm comm):
    try:
        sycl_queue.memcpy(dst, src, count).wait()
    except:
        mpi_xla_bridge.abort(0, comm, "Error: Unable to copy Data from the Device to CPU")
    return 

cdef inline void checked_sycl_queue_wait(queue* sycl_queue, MPI_Comm comm):
    try:
        sycl_queue.wait()
    except:
        mpi_xla_bridge.abort(0, comm, "Error: Unable to execute SYCL queue::wait()")
    return 



cdef void mpi_allgather_xpu(void* stream, void** buffers,
                            const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, sendtype_size, recvtype_size, comm_size
    cdef size_t sendbytes, recvbytes

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(AllgatherDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef AllgatherDescriptor* desc = <AllgatherDescriptor*>(opaque)
    cdef int sendcount = desc.sendcount
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int recvcount = desc.recvcount
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef MPI_Comm comm = desc.comm

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(sendtype, &sendtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        sendbytes = sendtype_size * sendcount
        in_buf = checked_malloc(sendbytes, comm)

        ierr = MPI_Type_size(recvtype, &recvtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        ierr = MPI_Comm_size(comm, &comm_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_size")

        # recvcount is received data *per process*
        recvbytes = recvtype_size * recvcount * comm_size
        out_buf = checked_malloc(recvbytes, comm)

        with gil:
            checked_sycl_memcpy(xqueue, in_buf, data , sendbytes, comm)

    mpi_xla_bridge.mpi_allgather(
        in_buf, sendcount, sendtype,
        out_buf, recvcount, recvtype,
        comm
    )

    if COPY_TO_HOST:
        # copy back to device
        with gil:
            checked_sycl_memcpy(xqueue, out_data, out_buf , recvbytes, comm)
        free(in_buf)
        free(out_buf)


cdef void mpi_allreduce_xpu(void* stream, void** buffers,
                            const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, dtype_size
    cdef size_t count

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(AllreduceDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef AllreduceDescriptor* desc = <AllreduceDescriptor*>(opaque)
    cdef int nitems = desc.nitems
    cdef MPI_Op op = desc.op
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        in_buf = checked_malloc(count, comm)
        out_buf = checked_malloc(count, comm)

        with gil:
            checked_sycl_memcpy(xqueue, in_buf, data , count, comm)

    mpi_xla_bridge.mpi_allreduce(in_buf, out_buf, nitems, dtype, op, comm)

    if COPY_TO_HOST:
        # copy back to device
        with gil:
            checked_sycl_memcpy(xqueue, out_data, out_buf , count, comm)
        free(in_buf)
        free(out_buf)


cdef void mpi_alltoall_xpu(void* stream, void** buffers,
                           const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, sendtype_size, recvtype_size, comm_size
    cdef size_t sendbytes, recvbytes

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(AlltoallDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef AlltoallDescriptor* desc = <AlltoallDescriptor*>(opaque)
    cdef int sendcount = desc.sendcount
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int recvcount = desc.recvcount
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef MPI_Comm comm = desc.comm

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host

        # counts are *per process*, so multiply with comm_size
        ierr = MPI_Comm_size(comm, &comm_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_size")

        ierr = MPI_Type_size(sendtype, &sendtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        sendbytes = sendtype_size * sendcount * comm_size
        in_buf = checked_malloc(sendbytes, comm)

        ierr = MPI_Type_size(recvtype, &recvtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        recvbytes = recvtype_size * recvcount * comm_size
        out_buf = checked_malloc(recvbytes, comm)

        with gil:
            checked_sycl_memcpy(xqueue, in_buf, data , sendbytes, comm)

    mpi_xla_bridge.mpi_alltoall(
        in_buf, sendcount, sendtype, out_buf, recvcount, recvtype, comm
    )

    if COPY_TO_HOST:
        # copy back to device
        with gil:
            checked_sycl_memcpy(xqueue, out_data, out_buf , recvbytes, comm)
        free(in_buf)
        free(out_buf)


cdef void mpi_barrier_xpu(void* stream, void** buffers,
                          const char* opaque, size_t opaque_len) nogil:
    if opaque_len != sizeof(BarrierDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef BarrierDescriptor* desc = <BarrierDescriptor*>(opaque)
    cdef MPI_Comm comm = desc.comm

    mpi_xla_bridge.mpi_barrier(comm)




cdef void mpi_bcast_xpu(void* stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, dtype_size, rank
    cdef size_t count

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* buf = out_data

    if opaque_len != sizeof(BcastDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef BcastDescriptor* desc = <BcastDescriptor*>(opaque)
    cdef int nitems = desc.nitems
    cdef int root = desc.root
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    ierr = MPI_Type_size(dtype, &dtype_size)
    mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")
    ierr = MPI_Comm_rank(comm, &rank)
    mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_rank")

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        count = dtype_size * nitems
        buf = checked_malloc(count, comm)
        with gil:
            checked_sycl_memcpy(xqueue, buf, data , count, comm)
    else:
        if rank == root:
            buf = data

    mpi_xla_bridge.mpi_bcast(buf, nitems, dtype, root, comm)

    if COPY_TO_HOST:
        if rank != root:
            # copy back to device
            with gil:
                checked_sycl_memcpy(xqueue, out_data, buf , count, comm)

        free(buf)




cdef void mpi_gather_xpu(void* stream, void** buffers,
                         const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, sendtype_size, recvtype_size, rank, size
    cdef size_t sendbytes, recvbytes

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(GatherDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef GatherDescriptor* desc = <GatherDescriptor*>(opaque)
    cdef int sendcount = desc.sendcount
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int recvcount = desc.recvcount
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef int root = desc.root
    cdef MPI_Comm comm = desc.comm

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(sendtype, &sendtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        sendbytes = sendtype_size * sendcount
        in_buf = checked_malloc(sendbytes, comm)

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

        out_buf = checked_malloc(recvbytes, comm)

        with gil:
            checked_sycl_memcpy(xqueue, in_buf, data , sendbytes, comm)

    mpi_xla_bridge.mpi_gather(
        in_buf, sendcount, sendtype, out_buf, recvcount, recvtype, root, comm
    )

    if COPY_TO_HOST:
        if rank == root:
            with gil:
                checked_sycl_memcpy(xqueue, out_data, out_buf , recvbytes, comm)
        free(in_buf)
        free(out_buf)


cdef void mpi_recv_xpu(void* stream, void** buffers,
                       const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, dtype_size
    cdef size_t count

    # decode inputs
    cdef void* out_buf = buffers[1]

    cdef void* recvbuf = out_buf

    if opaque_len != sizeof(RecvDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef RecvDescriptor* desc = <RecvDescriptor*>(opaque)
    cdef int nitems = desc.nitems
    cdef int source = desc.source
    cdef int tag = desc.tag
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype
    cdef MPI_Status* status = desc.status

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        recvbuf = checked_malloc(count, comm)

    mpi_xla_bridge.mpi_recv(recvbuf, nitems, dtype, source, tag, comm, status)

    if COPY_TO_HOST:
        with gil:
            checked_sycl_memcpy(xqueue, out_buf, recvbuf , count, comm)
        free(recvbuf)


cdef void mpi_reduce_xpu(void* stream, void** buffers,
                         const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, dtype_size, rank
    cdef size_t count

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(ReduceDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef ReduceDescriptor* desc = <ReduceDescriptor*>(opaque)
    cdef int nitems = desc.nitems
    cdef MPI_Op op = desc.op
    cdef int root = desc.root
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        in_buf = checked_malloc(count, comm)
        out_buf = checked_malloc(count, comm)
        with gil:
            checked_sycl_memcpy(xqueue, in_buf, data , count, comm)

    mpi_xla_bridge.mpi_reduce(in_buf, out_buf, nitems, dtype, op, root, comm)

    if COPY_TO_HOST:
        ierr = MPI_Comm_rank(comm, &rank)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_rank")

        if rank == root:
#            # copy back to device
            with gil:
                checked_sycl_memcpy(xqueue, out_data, out_buf , count, comm)

        free(in_buf)
        free(out_buf)


cdef void mpi_scan_xpu(void* stream, void** buffers,
                       const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, dtype_size
    cdef size_t count

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(ScanDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef ScanDescriptor* desc = <ScanDescriptor*>(opaque)
    cdef int nitems = desc.nitems
    cdef MPI_Op op = desc.op
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        in_buf = checked_malloc(count, comm)
        out_buf = checked_malloc(count, comm)
        with gil:
            checked_sycl_memcpy(xqueue, in_buf, data , count, comm)

    mpi_xla_bridge.mpi_scan(in_buf, out_buf, nitems, dtype, op, comm)

    if COPY_TO_HOST:
        # copy back to device
        with gil:
            checked_sycl_memcpy(xqueue, out_data, out_buf , count, comm)
        free(in_buf)
        free(out_buf)


cdef void mpi_scatter_xpu(void* stream, void** buffers,
                          const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, sendtype_size, recvtype_size, rank, size
    cdef size_t sendbytes, recvbytes

    # decode inputs
    cdef void* data = buffers[0]
    cdef void* out_data = buffers[2]

    cdef void* in_buf = data
    cdef void* out_buf = out_data

    if opaque_len != sizeof(ScatterDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef ScatterDescriptor* desc = <ScatterDescriptor*>(opaque)
    cdef int sendcount = desc.sendcount
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int recvcount = desc.recvcount
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef int root = desc.root
    cdef MPI_Comm comm = desc.comm

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

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

        in_buf = checked_malloc(sendbytes, comm)

        ierr = MPI_Type_size(recvtype, &recvtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        recvbytes = recvtype_size * recvcount
        out_buf = checked_malloc(recvbytes, comm)

        with gil:
            checked_sycl_memcpy(xqueue, in_buf, data , sendbytes, comm)

    mpi_xla_bridge.mpi_scatter(
        in_buf, sendcount, sendtype, out_buf, recvcount, recvtype, root, comm
    )

    if COPY_TO_HOST:
        # copy back to device
        with gil:
            checked_sycl_memcpy(xqueue, out_data, out_buf , recvbytes, comm)

        free(in_buf)
        free(out_buf)


cdef void mpi_send_xpu(void* stream, void** buffers,
                       const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, dtype_size
    cdef size_t count

    # decode inputs
    cdef void* data = buffers[0]

    cdef void* sendbuf = data

    if opaque_len != sizeof(SendDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef SendDescriptor* desc = <SendDescriptor*>(opaque)
    cdef int nitems = desc.nitems
    cdef int dest = desc.dest
    cdef int tag = desc.tag
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Datatype dtype = desc.dtype

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(dtype, &dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        sendbuf = checked_malloc(count, comm)

        with gil:
            checked_sycl_memcpy(xqueue, sendbuf, data , count, comm)

    mpi_xla_bridge.mpi_send(sendbuf, nitems, dtype, dest, tag, comm)

    if COPY_TO_HOST:
        free(sendbuf)


cdef void mpi_sendrecv_xpu(void* stream, void** buffers,
                           const char* opaque, size_t opaque_len) nogil:
    cdef int ierr, send_dtype_size, recv_dtype_size
    cdef size_t bytes_send, bytes_recv

    # decode inputs
    cdef void* in_buf = buffers[0]
    cdef void* out_buf = buffers[2]

    cdef void* sendbuf = in_buf
    cdef void* recvbuf = out_buf

    if opaque_len != sizeof(SendrecvDescriptor):
        with gil:
            raise RuntimeError("got wrong size of opaque argument")

    cdef SendrecvDescriptor* desc = <SendrecvDescriptor*>(opaque)
    cdef int sendcount = desc.sendcount
    cdef int dest = desc.dest
    cdef int sendtag = desc.sendtag
    cdef MPI_Datatype sendtype = desc.sendtype
    cdef int recvcount = desc.recvcount
    cdef int source = desc.source
    cdef int recvtag = desc.recvtag
    cdef MPI_Datatype recvtype = desc.recvtype
    cdef MPI_Comm comm = desc.comm
    cdef MPI_Status* status = desc.status

    cdef queue* xqueue = <queue*>stream
    with gil:
        checked_sycl_queue_wait(xqueue, comm) 

    if COPY_TO_HOST:
        # copy memory to host
        ierr = MPI_Type_size(sendtype, &send_dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        ierr = MPI_Type_size(recvtype, &recv_dtype_size)
        mpi_xla_bridge.abort_on_error(ierr, comm, u"Type_size")

        bytes_send = send_dtype_size * sendcount
        bytes_recv = recv_dtype_size * recvcount
        sendbuf = checked_malloc(bytes_send, comm)
        recvbuf = checked_malloc(bytes_recv, comm)

        with gil:
            checked_sycl_memcpy(xqueue, sendbuf, in_buf, bytes_send, comm)

    mpi_xla_bridge.mpi_sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    )

    if COPY_TO_HOST:
        with gil:
            checked_sycl_memcpy(xqueue, out_buf, recvbuf , bytes_recv, comm)
        free(recvbuf)


xpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    xpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)


register_custom_call_target(b"mpi_allgather", <void*>(mpi_allgather_xpu))
register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce_xpu))
register_custom_call_target(b"mpi_alltoall", <void*>(mpi_alltoall_xpu))
register_custom_call_target(b"mpi_barrier", <void*>(mpi_barrier_xpu))
register_custom_call_target(b"mpi_bcast", <void*>(mpi_bcast_xpu))
register_custom_call_target(b"mpi_gather", <void*>(mpi_gather_xpu))
register_custom_call_target(b"mpi_recv", <void*>(mpi_recv_xpu))
register_custom_call_target(b"mpi_reduce", <void*>(mpi_reduce_xpu))
register_custom_call_target(b"mpi_scan", <void*>(mpi_scan_xpu))
register_custom_call_target(b"mpi_scatter", <void*>(mpi_scatter_xpu))
register_custom_call_target(b"mpi_send", <void*>(mpi_send_xpu))
register_custom_call_target(b"mpi_sendrecv", <void*>(mpi_sendrecv_xpu))
