from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport int32_t, uint64_t
from libc.stdlib cimport malloc, free

from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Op,
    MPI_Datatype,
    MPI_Status,
    MPI_Type_size,
    MPI_Comm_rank
)

from .cuda_runtime_api cimport (
    cudaMemcpy,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaStream_t,
    cudaError_t,
    cudaMemcpyKind,
    cudaSuccess,
    cudaGetErrorName,
    cudaGetErrorString,
)

from . cimport mpi_xla_bridge
from .mpi_xla_bridge cimport abort_on_error


cpdef unicode py_string(char* c_str):
    py_str = <bytes> c_str
    return py_str.decode('UTF-8')

cpdef unicode GetErrorName(cudaError_t ierr):
    return py_string(cudaGetErrorName(ierr))

cpdef unicode GetErrorString(cudaError_t ierr):
    return py_string(cudaGetErrorString(ierr))


cdef bint COPY_TO_HOST = False

cpdef void set_copy_to_host(bint enable):
    global COPY_TO_HOST
    COPY_TO_HOST = enable


cdef inline void* checked_malloc(size_t count) nogil except *:
    cdef void* mem = malloc(count)
    if not mem:
        with gil:
            raise MemoryError(f"Failed to allocate {count} bytes")

    return mem


cdef inline cudaError_t checked_cuda_memcpy(void* dst, void* src, size_t count, cudaMemcpyKind kind) nogil except *:
    cdef cudaError_t ierr
    ierr = cudaMemcpy(dst, src, count, kind)

    if ierr != cudaSuccess:
        with gil:
            err_str = GetErrorName(ierr)
            err_des = GetErrorString(ierr)

            mystr = f"cudaMemcpy failed with the following error:\n\tError {ierr} {err_str}: {err_des}"
            raise RuntimeError(mystr)

    return ierr


#
# GPU XLA targets
#

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


cdef void mpi_allreduce(cudaStream_t* stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size
    cdef size_t count

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* token = buffers[1]
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
        abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        in_buf = checked_malloc(count)
        out_buf = checked_malloc(count)
        checked_cuda_memcpy(in_buf, data, count, cudaMemcpyDeviceToHost)

    mpi_xla_bridge.mpi_allreduce(in_buf, out_buf, nitems, dtype, op, comm, token)

    if COPY_TO_HOST:
        # copy back to device
        checked_cuda_memcpy(out_data, out_buf, count, cudaMemcpyHostToDevice)
        free(in_buf)
        free(out_buf)

    buffers[3] = token


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


cdef void mpi_bcast(cudaStream_t* stream, void** buffers,
                        const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size, rank
    cdef size_t count

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* token = buffers[1]
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
    abort_on_error(ierr, comm, u"Type_size")
    ierr = MPI_Comm_rank(comm, &rank)
    abort_on_error(ierr, comm, u"Comm_rank")

    count = dtype_size * nitems

    if COPY_TO_HOST:
        # copy memory to host
        buf = checked_malloc(count)
        if rank == root:
            checked_cuda_memcpy(buf, data, count, cudaMemcpyDeviceToHost)
    else:
        if rank == root:
            checked_cuda_memcpy(buf, data, count, cudaMemcpyDeviceToDevice)

    mpi_xla_bridge.mpi_bcast(buf, nitems, dtype, root, comm, token)

    if COPY_TO_HOST:
        # copy back to device
        checked_cuda_memcpy(out_data, buf, count, cudaMemcpyHostToDevice)
        free(buf)

    buffers[3] = token



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


cdef void mpi_send(cudaStream_t* stream, void** buffers,
                   const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size
    cdef size_t count

    #decode inputs
    cdef void* data = buffers[0]
    cdef void* token = buffers[1]

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
        abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        sendbuf = checked_malloc(count)
        checked_cuda_memcpy(sendbuf, data, count, cudaMemcpyDeviceToHost)

    mpi_xla_bridge.mpi_send(sendbuf, nitems, dtype, dest, tag, comm, token)

    if COPY_TO_HOST:
        free(sendbuf)

    buffers[2] = token


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


cdef void mpi_recv(cudaStream_t* stream, void** buffers,
                   const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, dtype_size
    cdef size_t count

    #decode inputs
    cdef void* token = buffers[0]
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
        abort_on_error(ierr, comm, u"Type_size")

        count = dtype_size * nitems
        recvbuf = checked_malloc(count)

    mpi_xla_bridge.mpi_recv(recvbuf, nitems, dtype, source, tag, comm, status, token)

    if COPY_TO_HOST:
        checked_cuda_memcpy(out_buf, recvbuf, count, cudaMemcpyHostToDevice)
        free(recvbuf)

    buffers[2] = token


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


cdef void mpi_sendrecv(cudaStream_t* stream, void** buffers,
                       const char* opaque, size_t opaque_len) nogil except *:
    cdef int ierr, send_dtype_size, recv_dtype_size
    cdef size_t bytes_send, bytes_recv

    #decode inputs
    cdef void* in_buf = buffers[0]
    cdef void* token = buffers[1]
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
        abort_on_error(ierr, comm, u"Type_size")

        ierr = MPI_Type_size(recvtype, &recv_dtype_size)
        abort_on_error(ierr, comm, u"Type_size")

        bytes_send = send_dtype_size * sendcount
        bytes_recv = recv_dtype_size * recvcount
        sendbuf = checked_malloc(bytes_send)
        recvbuf = checked_malloc(bytes_recv)
        checked_cuda_memcpy(sendbuf, in_buf, bytes_send, cudaMemcpyDeviceToHost)

    mpi_xla_bridge.mpi_sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status, token
    )

    if COPY_TO_HOST:
        checked_cuda_memcpy(out_buf, recvbuf, bytes_recv, cudaMemcpyHostToDevice)
        free(recvbuf)

    buffers[3] = token


gpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    gpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)


register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce))
register_custom_call_target(b"mpi_bcast", <void*>(mpi_bcast))
register_custom_call_target(b"mpi_send", <void*>(mpi_send))
register_custom_call_target(b"mpi_recv", <void*>(mpi_recv))
register_custom_call_target(b"mpi_sendrecv", <void*>(mpi_sendrecv))
