from mpi4py.libmpi cimport MPI_Comm, MPI_Datatype, MPI_Status, MPI_Op

cdef int abort(int ierr, MPI_Comm comm, unicode message) nogil

cdef int abort_on_error(int ierr, MPI_Comm comm, unicode mpi_op) nogil

cdef void mpi_allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                        void* recvbuf, int recvcount, MPI_Datatype recvtype,
                        MPI_Comm comm) nogil

cdef void mpi_allreduce(void* sendbuf, void* recvbuf, int nitems,
                        MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) nogil

cdef void mpi_alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                       void* recvbuf, int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm) nogil

cdef void mpi_barrier(MPI_Comm comm) nogil

cdef void mpi_bcast(void* sendrecvbuf, int nitems, MPI_Datatype dtype,
                   int root, MPI_Comm comm) nogil

cdef void mpi_gather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root, MPI_Comm comm) nogil

cdef void mpi_recv(void* recvbuf, int nitems, MPI_Datatype dtype, int source,
                   int tag, MPI_Comm comm, MPI_Status* status) nogil

cdef void mpi_reduce(void* sendbuf, void* recvbuf, int nitems,
                     MPI_Datatype dtype, MPI_Op op, int root,
                     MPI_Comm comm) nogil

cdef void mpi_scan(void* sendbuf, void* recvbuf, int nitems,
                   MPI_Datatype dtype, MPI_Op op, MPI_Comm comm) nogil

cdef void mpi_scatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                      void* recvbuf, int recvcount, MPI_Datatype recvtype,
                      int root, MPI_Comm comm) nogil

cdef void mpi_send(void* sendbuf, int nitems, MPI_Datatype dtype,
                   int destination, int tag, MPI_Comm comm) nogil

cdef void mpi_sendrecv(void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                       void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                       MPI_Comm comm, MPI_Status* status) nogil
