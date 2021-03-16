import sys

from libc.stdint cimport uintptr_t

from mpi4py.libmpi cimport (
    MPI_Abort,
    MPI_Comm,
    MPI_Comm_rank,
    MPI_Datatype,
    MPI_Op,
    MPI_Status,
    MPI_STATUS_IGNORE,
    MPI_SUCCESS,
)

cimport mpi4py.libmpi as libmpi

# MPI_STATUS_IGNORE is not exposed to Python by mpi4py, so we
# export its memory address as Python int here.
# This can then be passed to all functions that expect
# MPI_Status* instead of a pointer to a real status object.
MPI_STATUS_IGNORE_ADDR = int(<uintptr_t>MPI_STATUS_IGNORE)


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

cdef inline int abort(int ierr, MPI_Comm comm, unicode message) nogil:
    with gil:
        sys.stderr.write(message)
        sys.stderr.flush()

    return MPI_Abort(comm, ierr)


cdef inline int abort_on_error(int ierr, MPI_Comm comm, unicode mpi_op) nogil:
    if ierr == MPI_SUCCESS:
        return 0

    cdef int rank
    MPI_Comm_rank(comm, &rank)

    with gil:
        message = f'r{rank} | MPI_{mpi_op} returned error code {ierr} - aborting\n'

    return abort(ierr, comm, message)


#
# Wrapped MPI primitives
#

cdef void mpi_allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                        void* recvbuf, int recvcount, MPI_Datatype recvtype,
                        MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(
                f"MPI_Allgather sending {sendcount}, receiving {recvcount} items",
                comm
            )

    # MPI Call
    ierr = libmpi.MPI_Allgather(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm
    )
    abort_on_error(ierr, comm, u"Allgather")


cdef void mpi_allreduce(void* sendbuf, void* recvbuf, int nitems,
                        MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Allreduce with {nitems} items", comm)

    # MPI Call
    ierr = libmpi.MPI_Allreduce(sendbuf, recvbuf, nitems, dtype, op, comm)
    abort_on_error(ierr, comm, u"Allreduce")


cdef void mpi_alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                       void* recvbuf, int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(
                f"MPI_Alltoall sending {sendcount}, receiving {recvcount} items",
                comm
            )

    # MPI Call
    ierr = libmpi.MPI_Alltoall(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm
    )
    abort_on_error(ierr, comm, u"Alltoall")


cdef void mpi_bcast(void* sendrecvbuf, int nitems, MPI_Datatype dtype,
                    int root, MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Bcast -> {root} with {nitems} items", comm)

    # MPI Call
    ierr = libmpi.MPI_Bcast(sendrecvbuf, nitems, dtype, root, comm)
    abort_on_error(ierr, comm, u"Bcast")


cdef void mpi_gather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root, MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(
                f"MPI_Gather -> {root} sending {sendcount}, "
                f"receiving {recvcount} items", comm
            )

    # MPI Call
    ierr = libmpi.MPI_Gather(
        sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm
    )
    abort_on_error(ierr, comm, u"Gather")


cdef void mpi_recv(void* recvbuf, int nitems, MPI_Datatype dtype, int source,
                   int tag, MPI_Comm comm, MPI_Status* status) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Recv <- {source} with tag {tag} and {nitems} items", comm)

    # MPI Call
    ierr = libmpi.MPI_Recv(recvbuf, nitems, dtype, source, tag, comm, status)
    abort_on_error(ierr, comm, u"Recv")


cdef void mpi_reduce(void* sendbuf, void* recvbuf, int nitems,
                     MPI_Datatype dtype, MPI_Op op, int root,
                     MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Reduce -> {root} with {nitems} items", comm)

    # MPI Call
    ierr = libmpi.MPI_Reduce(sendbuf, recvbuf, nitems, dtype, op, root, comm)
    abort_on_error(ierr, comm, u"Reduce")


cdef void mpi_scan(void* sendbuf, void* recvbuf, int nitems,
                   MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(f"MPI_Scan with {nitems} items", comm)

    # MPI Call
    ierr = libmpi.MPI_Scan(sendbuf, recvbuf, nitems, dtype, op, comm)
    abort_on_error(ierr, comm, u"Scan")


cdef void mpi_scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                      void* recvbuf, int recvcount, MPI_Datatype recvtype,
                      int root, MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(
                f"MPI_Scatter -> {root} sending {sendcount}, "
                f"receiving {recvcount} items", comm
            )

    # MPI Call
    ierr = libmpi.MPI_Scatter(
        sendbuf, sendcount, sendtype,
        recvbuf, recvcount, recvtype,
        root, comm
    )
    abort_on_error(ierr, comm, u"Scatter")


cdef void mpi_send(void* sendbuf, int nitems, MPI_Datatype dtype,
                   int destination, int tag, MPI_Comm comm) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(
                f"MPI_Send -> {destination} with tag {tag} and {nitems} items", comm
            )

    # MPI Call
    ierr = libmpi.MPI_Send(sendbuf, nitems, dtype, destination, tag, comm)
    abort_on_error(ierr, comm, u"Send")


cdef void mpi_sendrecv(
    void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
    void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
    MPI_Comm comm, MPI_Status* status
) nogil:
    cdef int ierr

    if PRINT_DEBUG:
        with gil:
            print_debug(
                f"MPI_Sendrecv <- {source} (tag {recvtag}, {recvcount} items) / "
                f"-> {dest} (tag {sendtag}, {sendcount} items) ", comm
            )

    # MPI Call
    ierr = libmpi.MPI_Sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    )
    abort_on_error(ierr, comm, u"Sendrecv")
