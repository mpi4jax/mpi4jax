# cython: language_level=2
# distutils: language = c++

cimport mpi4py.MPI as MPI
cimport mpi4py.libmpi as libmpi
from mpi4py.libmpi cimport MPI_Comm, MPI_Op, MPI_Datatype

from cpython.pycapsule cimport PyCapsule_New

from libc.stdio cimport printf
from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t

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

register_custom_call_target(b"mpi_allreduce", <void*>(mpi_allreduce))
