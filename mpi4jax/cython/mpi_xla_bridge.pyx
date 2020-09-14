# cython: language_level=3

import sys

from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport int32_t, int64_t, uint64_t

cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Op,
    MPI_Datatype,
    MPI_Status,
    MPI_STATUS_IGNORE,
    MPI_SUCCESS,
    MPI_Comm_rank,
)


# MPI_STATUS_IGNORE is not exposed to Python by mpi4py, so we
# export its memory address as Python int here.
# This can then be passed to all functions that expect
# MPI_Status* instead of a pointer to a real status object.
MPI_STATUS_IGNORE_ADDR = int(<uint64_t>MPI_STATUS_IGNORE)


#
# Error handling
#

cdef inline int abort_on_error(int ierr, MPI_Comm comm, unicode mpi_op) nogil:
    if ierr == MPI_SUCCESS:
        return 0

    cdef int rank
    MPI_Comm_rank(comm, &rank)

    with gil:
        sys.stderr.write(
            f'r{rank} | MPI_{mpi_op} returned error code {ierr} - aborting\n'
        )
        sys.stderr.flush()
        return libmpi.MPI_Abort(comm, ierr)


#
# Logging
#

cdef bint PRINT_DEBUG = False

cpdef void set_logging(bint enable):
    global PRINT_DEBUG
    PRINT_DEBUG = enable


#
# Custom XLA targets (MPI primitives)
#

cdef void mpi_send(void** out_ptr, void** data_ptr) nogil:
    cdef int rank, ierr

    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t destination = (<int32_t*>(data_ptr[2]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[3]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[5]))[0])
    cdef void* token = data_ptr[6]

    if PRINT_DEBUG:
        MPI_Comm_rank(comm, &rank)
        with gil:
            print(f"r{rank} | MPI_Send -> {destination} with tag {tag} and token {<uint64_t>token:x}")

    # MPI Call
    ierr = libmpi.MPI_Send(data_ptr[1], nitems, dtype, destination, tag, comm)
    abort_on_error(ierr, comm, u"Send")

    out_ptr[0] = token


cdef void mpi_recv(void** out_ptr, void** data_ptr) nogil:
    cdef int rank, ierr

    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t source = (<int32_t*>(data_ptr[1]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Status* status = <MPI_Status*>((<uint64_t*>(data_ptr[5]))[0])
    cdef void* token = data_ptr[6]

    if PRINT_DEBUG:
        MPI_Comm_rank(comm, &rank)
        with gil:
            print(f"r{rank} | MPI_Recv <- {source} with tag {tag} and token {<uint64_t> token:x}")

    # MPI Call
    ierr = libmpi.MPI_Recv(out_ptr[0], nitems, dtype, source, tag, comm, status)
    abort_on_error(ierr, comm, u"Recv")

    out_ptr[1] = token


cdef void mpi_sendrecv(void** out_ptr, void** data_ptr) nogil:
    cdef int rank

    #decode inputs
    cdef int32_t sendcount = (<int32_t*>(data_ptr[0]))[0]
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

    if PRINT_DEBUG:
        MPI_Comm_rank(comm, &rank)
        with gil:
            print(
                f"r{rank} | MPI_Sendrecv <- {source} (tag {recvtag}) / -> {dest} (tag {sendtag}) "
                f"with token {<uint64_t>token:x}"
            )

    # MPI Call
    ierr = libmpi.MPI_Sendrecv(
        data_ptr[1], sendcount, sendtype, dest, sendtag,
        out_ptr[0], recvcount, recvtype, source, recvtag,
        comm, status
    )
    abort_on_error(ierr, comm, u"Sendrecv")

    out_ptr[1] = token


cdef void mpi_allreduce(void** out_ptr, void** data_ptr) nogil:
    cdef int rank, ierr

    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef MPI_Op op = <MPI_Op>((<uint64_t*>(data_ptr[2]))[0])
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef void* token = data_ptr[5]

    if PRINT_DEBUG:
        MPI_Comm_rank(comm, &rank)
        with gil:
            print(f"r{rank} | MPI_Allreduce with token {<uint64_t>token:x}")

    # MPI Call
    ierr = libmpi.MPI_Allreduce(data_ptr[1], out_ptr[0], nitems, dtype, op, comm)
    abort_on_error(ierr, comm, u"Allreduce")

    out_ptr[1] = token


cpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"mpi_send", <void*>(mpi_send))
register_custom_call_target(b"mpi_recv", <void*>(mpi_recv))
register_custom_call_target(b"mpi_sendrecv", <void*>(mpi_sendrecv))
register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce))
