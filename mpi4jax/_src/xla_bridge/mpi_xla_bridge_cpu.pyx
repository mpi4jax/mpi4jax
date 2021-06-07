from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport uintptr_t
from libc.string cimport memcpy

from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Comm_rank,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
)

from . cimport mpi_xla_bridge


#
# CPU XLA targets
#
cdef void mpi_barrier_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[0]))[0])
    mpi_xla_bridge.mpi_barrier(comm)


cdef void mpi_allgather_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int sendcount = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef MPI_Datatype sendtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[2]))[0])
    cdef int recvcount = (<int*>(data_ptr[3]))[0]
    cdef MPI_Datatype recvtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[5]))[0])

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_allgather(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm
    )


cdef void mpi_allreduce_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int nitems = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef MPI_Op op = <MPI_Op>((<uintptr_t*>(data_ptr[2]))[0])
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_allreduce(
        sendbuf, recvbuf, nitems, dtype, op, comm
    )


cdef void mpi_alltoall_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int sendcount = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef MPI_Datatype sendtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[2]))[0])
    cdef int recvcount = (<int*>(data_ptr[3]))[0]
    cdef MPI_Datatype recvtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[5]))[0])

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_alltoall(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm
    )


cdef void mpi_bcast_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int nitems = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef int root = (<int*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])

    cdef void* recvbuf = out_ptr[0]

    cdef int ierr, rank
    cdef void* buf
    ierr = MPI_Comm_rank(comm, &rank)
    mpi_xla_bridge.abort_on_error(ierr, comm, u"Comm_rank")

    if rank == root:
        buf = sendbuf
    else:
        buf = recvbuf

    mpi_xla_bridge.mpi_bcast(buf, nitems, dtype, root, comm)


cdef void mpi_gather_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int sendcount = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef MPI_Datatype sendtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[2]))[0])
    cdef int recvcount = (<int*>(data_ptr[3]))[0]
    cdef MPI_Datatype recvtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])
    cdef int root = (<int*>(data_ptr[5]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[6]))[0])

    cdef void* recvbuf = out_ptr[0]

    mpi_xla_bridge.mpi_gather(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm
    )


cdef void mpi_recv_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int nitems = (<int*>(data_ptr[0]))[0]
    cdef int source = (<int*>(data_ptr[1]))[0]
    cdef int tag = (<int*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])
    cdef MPI_Status* status = <MPI_Status*>((<uintptr_t*>(data_ptr[5]))[0])

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_recv(recvbuf, nitems, dtype, source, tag, comm, status)


cdef void mpi_reduce_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int nitems = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef MPI_Op op = <MPI_Op>((<uintptr_t*>(data_ptr[2]))[0])
    cdef int root = (<int*>(data_ptr[3]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[4]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[5]))[0])

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_reduce(
        sendbuf, recvbuf, nitems, dtype, op, root, comm
    )


cdef void mpi_scan_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int nitems = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef MPI_Op op = <MPI_Op>((<uintptr_t*>(data_ptr[2]))[0])
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_scan(
        sendbuf, recvbuf, nitems, dtype, op, comm
    )


cdef void mpi_scatter_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int sendcount = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef MPI_Datatype sendtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[2]))[0])
    cdef int recvcount = (<int*>(data_ptr[3]))[0]
    cdef MPI_Datatype recvtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])
    cdef int root = (<int*>(data_ptr[5]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[6]))[0])

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_scatter(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm
    )


cdef void mpi_send_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int nitems = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef int destination = (<int*>(data_ptr[2]))[0]
    cdef int tag = (<int*>(data_ptr[3]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[4]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[5]))[0])

    mpi_xla_bridge.mpi_send(sendbuf, nitems, dtype, destination, tag, comm)


cdef void mpi_sendrecv_cpu(void** out_ptr, void** data_ptr) nogil:
    cdef int sendcount = (<int*>(data_ptr[0]))[0]
    cdef void* sendbuf = data_ptr[1]
    cdef int dest = (<int*>(data_ptr[2]))[0]
    cdef int sendtag = (<int*>(data_ptr[3]))[0]
    cdef MPI_Datatype sendtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[4]))[0])

    cdef int recvcount = (<int*>(data_ptr[5]))[0]
    cdef int source = (<int*>(data_ptr[6]))[0]
    cdef int recvtag = (<int*>(data_ptr[7]))[0]
    cdef MPI_Datatype recvtype = <MPI_Datatype>((<uintptr_t*>(data_ptr[8]))[0])

    cdef MPI_Comm comm = <MPI_Comm>((<uintptr_t*>(data_ptr[9]))[0])
    cdef MPI_Status* status = <MPI_Status*>((<uintptr_t*>(data_ptr[10]))[0])

    cdef void* recvbuf = out_ptr[0]
    mpi_xla_bridge.mpi_sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    )


cpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"mpi_allgather", <void*>(mpi_allgather_cpu))
register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce_cpu))
register_custom_call_target(b"mpi_alltoall", <void*>(mpi_alltoall_cpu))
register_custom_call_target(b"mpi_barrier", <void*>(mpi_barrier_cpu))
register_custom_call_target(b"mpi_bcast", <void*>(mpi_bcast_cpu))
register_custom_call_target(b"mpi_gather", <void*>(mpi_gather_cpu))
register_custom_call_target(b"mpi_recv", <void*>(mpi_recv_cpu))
register_custom_call_target(b"mpi_reduce", <void*>(mpi_reduce_cpu))
register_custom_call_target(b"mpi_scan", <void*>(mpi_scan_cpu))
register_custom_call_target(b"mpi_scatter", <void*>(mpi_scatter_cpu))
register_custom_call_target(b"mpi_send", <void*>(mpi_send_cpu))
register_custom_call_target(b"mpi_sendrecv", <void*>(mpi_sendrecv_cpu))
