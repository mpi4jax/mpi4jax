import sys

from libc.stdint cimport int32_t, uint64_t

cimport mpi4py.libmpi as libmpi
from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Op,
    MPI_Datatype,
    MPI_Status,
    MPI_STATUS_IGNORE,
    MPI_SUCCESS,
    MPI_Comm_rank,
    MPI_Abort,
)

# MPI_STATUS_IGNORE is not exposed to Python by mpi4py, so we
# export its memory address as Python int here.
# This can then be passed to all functions that expect
# MPI_Status* instead of a pointer to a real status object.
MPI_STATUS_IGNORE_ADDR = int(<uint64_t>MPI_STATUS_IGNORE)


#
# Logging
#

cdef bint PRINT_DEBUG = False


cpdef void set_logging(bint enable):
    global PRINT_DEBUG
    PRINT_DEBUG = enable


cdef inline void print_debug(unicode message, MPI_Comm comm) nogil:
    cdef int rank
    MPI_Comm_rank(comm, &rank)
    with gil:
        print(f"r{rank} | {message}")


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
        return MPI_Abort(comm, ierr)


#
# Wrapped MPI primitives
#

cdef void mpi_send(void* sendbuf, int32_t nitems, MPI_Datatype dtype,
                   int32_t destination, int32_t tag, MPI_Comm comm, void* token) nogil:
    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Send -> {destination} with tag {tag} and token {<uint64_t>token:x}", comm)

    # MPI Call
    cdef int ierr
    ierr = libmpi.MPI_Send(sendbuf, nitems, dtype, destination, tag, comm)
    abort_on_error(ierr, comm, u"Send")


cdef void mpi_recv(void* recvbuf, int32_t nitems, MPI_Datatype dtype, int32_t source,
                   int32_t tag, MPI_Comm comm, MPI_Status* status, void* token) nogil:
    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Recv <- {source} with tag {tag} and token {<uint64_t> token:x}", comm)

    # MPI Call
    cdef int ierr
    ierr = libmpi.MPI_Recv(recvbuf, nitems, dtype, source, tag, comm, status)
    abort_on_error(ierr, comm, u"Recv")


cdef void mpi_sendrecv(void* sendbuf, int32_t sendcount, MPI_Datatype sendtype, int32_t dest, int32_t sendtag,
                       void* recvbuf, int32_t recvcount, MPI_Datatype recvtype, int32_t source, int32_t recvtag,
                       MPI_Comm comm, MPI_Status* status, void* token) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(
                f"MPI_Sendrecv <- {source} (tag {recvtag}) / -> {dest} (tag {sendtag}) "
                f"with token {<uint64_t>token:x}", comm
            )

    # MPI Call
    ierr = libmpi.MPI_Sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    )
    abort_on_error(ierr, comm, u"Sendrecv")


cdef void mpi_bcast(void* sendrecvbuf, int32_t nitems, MPI_Datatype dtype,
                   int32_t root, MPI_Comm comm, void* token) nogil:
    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Bcast -> {root} with token {<uint64_t>token:x}", comm)

    # MPI Call
    cdef int ierr
    ierr = libmpi.MPI_Bcast(sendrecvbuf, nitems, dtype, root, comm)
    abort_on_error(ierr, comm, u"Bcast")


cdef void mpi_allreduce(void* sendbuf, void* recvbuf, int32_t nitems,
                        MPI_Datatype dtype, MPI_Op op, MPI_Comm comm, void* token) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Allreduce with token {<uint64_t>token:x}", comm)

    # MPI Call
    ierr = libmpi.MPI_Allreduce(sendbuf, recvbuf, nitems, dtype, op, comm)
    abort_on_error(ierr, comm, u"Allreduce")
