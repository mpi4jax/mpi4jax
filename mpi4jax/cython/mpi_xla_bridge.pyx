# cython: language_level=2
# distutils: language = c++

DEF DEBUG = False

cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py.libmpi cimport (
    MPI_Comm,
    MPI_Op,
    MPI_Datatype,
    MPI_Status,
    MPI_STATUS_IGNORE,
    MPI_SUCCESS,
)

from cpython.pycapsule cimport PyCapsule_New

from libc.stdio cimport printf
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t

# -----------------------------------------------------------------------------
# Error checking
# From https://github.com/mpi4py/mpi4py/blob/3285cf9a06ed1916966a9e2a0af05defe1c3a40e/src/mpi4py/MPI/atimport.pxi


cdef extern from "Python.h":
    void PyErr_SetObject(object, object)
    void *PyExc_RuntimeError
    void *PyExc_NotImplementedError


cdef object MPIException = <object>PyExc_RuntimeError


cdef int PyMPI_Raise(int ierr) except -1 with gil:
    if (<void*>MPIException) != NULL:
        PyErr_SetObject(MPIException, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
    return 0


cdef inline int CHKERR(int ierr) nogil except -1:
    if ierr == MPI_SUCCESS: return 0
    PyMPI_Raise(ierr)
    return -1

# -----------------------------------------------------------------------------


cdef void mpi_send(int* out_ptr, void** data_ptr) nogil:
    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t destination = (<int32_t*>(data_ptr[2]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[3]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[5]))[0])

    IF DEBUG:
        printf("MPI_Send to %d with tag %d\n", destination, tag)

    # MPI Call
    CHKERR(libmpi.MPI_Send(data_ptr[1], nitems, dtype, destination, tag, comm))

    out_ptr[0] = 0


cdef void mpi_recv(void* out_ptr, void** data_ptr) nogil:
    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t source = (<int32_t*>(data_ptr[1]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Status* status = <MPI_Status*>((<uint64_t*>(data_ptr[5]))[0])

    IF DEBUG:
        printf("MPI_Recv from %d with tag %d\n", source, tag)

    # MPI Call
    CHKERR(libmpi.MPI_Recv(out_ptr, nitems, dtype, source, tag, comm, status))


cdef void mpi_recv_ignore_status(void* out_ptr, void** data_ptr) nogil:
    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t source = (<int32_t*>(data_ptr[1]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Status* status = MPI_STATUS_IGNORE

    IF DEBUG:
        printf("MPI_Recv from %d with tag %d\n", source, tag)

    # MPI Call
    CHKERR(libmpi.MPI_Recv(out_ptr, nitems, dtype, source, tag, comm, status))


cdef void mpi_sendrecv(void* out_ptr, void** data_ptr) nogil:
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

    IF DEBUG:
        printf("MPI_Sendrecv from %d (tag %d) to %d (tag %d)\n", source, recvtag, dest, sendtag)

    # MPI Call
    CHKERR(libmpi.MPI_Sendrecv(
        data_ptr[1], sendcount, sendtype, dest, sendtag,
        out_ptr, recvcount, recvtype, source, recvtag,
        comm, status
    ))



cdef void mpi_sendrecv_ignore_status(void* out_ptr, void** data_ptr) nogil:
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
    cdef MPI_Status* status = MPI_STATUS_IGNORE

    IF DEBUG:
        printf("MPI_Sendrecv from %d (tag %d) to %d (tag %d)\n", source, recvtag, dest, sendtag)

    # MPI Call
    CHKERR(libmpi.MPI_Sendrecv(
        data_ptr[1], sendcount, sendtype, dest, sendtag,
        out_ptr, recvcount, recvtype, source, recvtag,
        comm, status
    ))


cdef void mpi_allreduce(void* out_ptr, void** data_ptr) nogil:
    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef MPI_Op op = <MPI_Op>((<uint64_t*>(data_ptr[2]))[0])
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])

    IF DEBUG:
        printf("MPI_Allreduce")

    # MPI Call
    CHKERR(libmpi.MPI_Allreduce(data_ptr[1], out_ptr, nitems, dtype, op, comm))


cpu_custom_call_targets = {}


cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"mpi_send", <void*>(mpi_send))
register_custom_call_target(b"mpi_recv", <void*>(mpi_recv))
register_custom_call_target(b"mpi_recv_ignore_status", <void*>(mpi_recv_ignore_status))
register_custom_call_target(b"mpi_sendrecv", <void*>(mpi_sendrecv))
register_custom_call_target(b"mpi_sendrecv_ignore_status", <void*>(mpi_sendrecv_ignore_status))
register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce))
