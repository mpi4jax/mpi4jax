# cython: language_level=2
# distutils: language = c++

cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py.libmpi cimport MPI_Comm, MPI_Op, MPI_Datatype, MPI_Status, MPI_STATUS_IGNORE, MPI_Type_size

from cpython.pycapsule cimport PyCapsule_New

from libc.stdio cimport printf
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from libc.string cimport memcpy


cdef void mpi_send(int* out_ptr, void** data_ptr) nogil:
    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t destination = (<int32_t*>(data_ptr[2]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[3]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[5]))[0])

    # MPI Call
    libmpi.MPI_Send(data_ptr[1], nitems, dtype, destination, tag, comm)

    out_ptr[0] = 0


cdef void mpi_recv(void* out_ptr, void** data_ptr) nogil:
    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t source = (<int32_t*>(data_ptr[1]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Status* status = <MPI_Status*>((<uint64_t*>(data_ptr[5]))[0])

    # MPI Call
    libmpi.MPI_Recv(out_ptr, nitems, dtype, source, tag, comm, status)


cdef void mpi_recv_ignore_status(void* out_ptr, void** data_ptr) nogil:
    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef int32_t source = (<int32_t*>(data_ptr[1]))[0]
    cdef int32_t tag = (<int32_t*>(data_ptr[2]))[0]
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])
    cdef MPI_Status* status = MPI_STATUS_IGNORE

    # MPI Call
    libmpi.MPI_Recv(out_ptr, nitems, dtype, source, tag, comm, status)


cdef void mpi_allreduce(void* out_ptr, void** data_ptr) nogil:
    #decode inputs
    cdef int32_t nitems = (<int32_t*>(data_ptr[0]))[0]
    cdef MPI_Op op = <MPI_Op>((<uint64_t*>(data_ptr[2]))[0])
    cdef MPI_Comm comm = <MPI_Comm>((<uint64_t*>(data_ptr[3]))[0])
    cdef MPI_Datatype dtype = <MPI_Datatype>((<uint64_t*>(data_ptr[4]))[0])

    # MPI Call
    libmpi.MPI_Allreduce(data_ptr[1], out_ptr, nitems, dtype, op, comm)


cpu_custom_call_targets = {}


cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"mpi_send", <void*>(mpi_send))
register_custom_call_target(b"mpi_recv", <void*>(mpi_recv))
register_custom_call_target(b"mpi_recv_ignore_status", <void*>(mpi_recv_ignore_status))
register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce))
