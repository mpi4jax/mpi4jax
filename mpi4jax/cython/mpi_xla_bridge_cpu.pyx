from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport int32_t, uint64_t
from libc.string cimport memcpy

from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
    MPI_Comm_rank,
    MPI_Type_size,
)

from .mpi_xla_bridge cimport abort_on_error

from . cimport mpi_xla_bridge


#
# CPU XLA targets
#

cdef void mpi_send_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef int32_t destination = (<int32_t*>(data_ptr[2]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[3]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[5]))[0])
    cdef void* token = data_ptr[6]

    mpi_xla_bridge.mpi_send(sendbuf, nitems, dtype, destination, tag, comm, token)
    out_ptr[0] = token


cdef void mpi_recv_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t source = (<int32_t*>(data_ptr[1]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Status* status = <MPI_Status*>((<uint64_t*>(data_ptr[5]))[0])
    cdef void* token = data_ptr[6]

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_recv(recvbuf, nitems, dtype, source, tag, comm, status, token)
    out_ptr[1] = token


cdef void mpi_sendrecv_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int32_t sendcount = (<int32_t*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef int32_t dest = (<int32_t*>(data_ptr[2]))[0]
    cdef int32_t sendtag = (<int32_t*>(data_ptr[3]))[0]
    cdef MPI_Datatype sendtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])

    cdef int32_t recvcount = (<int32_t*>(data_ptr[5]))[0]
    cdef int32_t source = (<int32_t*>(data_ptr[6]))[0]
    cdef int32_t recvtag = (<int32_t*>(data_ptr[7]))[0]
    cdef MPI_Datatype recvtype = <MPI_Datatype>((<uint64_t*>(data_ptr[8]))[0])

    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[9]))[0])
    cdef MPI_Status* status = <MPI_Status*>((<uint64_t*>(data_ptr[10]))[0])

    cdef void* token = data_ptr[11]

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status, token
    )
    out_ptr[1] = token


cdef void mpi_bcast_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef void* send_buf = data_ptr[1]
    cdef int32_t root = (<int32_t*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef void* token = data_ptr[5]

    cdef void* out_buf = out_ptr[0]

    cdef int dtype_size, rank
    ierr = MPI_Type_size(dtype, &dtype_size)
    abort_on_error(ierr, comm, u"Type_size")
    ierr = MPI_Comm_rank(comm, &rank)
    abort_on_error(ierr, comm, u"Comm_rank")

    if rank == root:
        count = dtype_size * nitems
        memcpy(out_buf, send_buf, count)

    mpi_xla_bridge.mpi_bcast(out_buf, nitems, dtype, root, comm, token)
    out_ptr[1] = token


cdef void mpi_allreduce_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef MPI_Op op = <MPI_Op>((<uint64_t*>(data_ptr[2]))[0])
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef void* token = data_ptr[5]

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_allreduce(
        sendbuf, recvbuf, nitems, dtype, op, comm, token
    )
    out_ptr[1] = token


cpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"mpi_send", <void*>(mpi_send_cpu))
register_custom_call_target(b"mpi_recv", <void*>(mpi_recv_cpu))
register_custom_call_target(b"mpi_sendrecv", <void*>(mpi_sendrecv_cpu))
register_custom_call_target(b"mpi_bcast", <void*>(mpi_bcast_cpu))
register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce_cpu))
