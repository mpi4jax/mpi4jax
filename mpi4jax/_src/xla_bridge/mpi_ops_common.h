// Common MPI operations and utilities shared between CPU and CUDA backends
// This header provides:
// - MPI handle conversion utilities (from_handle/to_handle)
// - Error handling utilities
// - Debug logging utilities
// - Core MPI wrapper functions

#ifndef MPI4JAX_MPI_OPS_COMMON_H_
#define MPI4JAX_MPI_OPS_COMMON_H_

#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <chrono>
#include <random>
#include <string>

namespace mpi4jax {

// ============================================================================
// MPI handle conversion utilities
// ============================================================================
// MPI implementations differ in how they define handle types:
// - OpenMPI: MPI_Comm, MPI_Datatype, MPI_Op are pointers (e.g., ompi_communicator_t*)
// - MPICH: MPI_Comm, MPI_Datatype, MPI_Op are integers (int)
// We use memcpy for type-safe conversion that works for both cases.

template<typename MPI_Handle>
inline MPI_Handle from_handle(int64_t handle) {
    MPI_Handle result;
    std::memcpy(&result, &handle, sizeof(result));
    return result;
}

template<typename MPI_Handle>
inline int64_t to_handle(MPI_Handle handle) {
    int64_t result = 0;
    std::memcpy(&result, &handle, sizeof(handle));
    return result;
}

// ============================================================================
// Error handling
// ============================================================================
inline int abort_on_error(int ierr, MPI_Comm comm, const char* mpi_op) {
    if (ierr == MPI_SUCCESS) {
        return 0;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    char error_string[MPI_MAX_ERROR_STRING];
    int length;
    MPI_Error_string(ierr, error_string, &length);

    std::cerr << "r" << rank << " | MPI_" << mpi_op
              << " returned error code " << ierr << ": "
              << std::string(error_string, length) << " - aborting" << std::endl;
    std::cerr.flush();

    return MPI_Abort(comm, ierr);
}

// ============================================================================
// Logging utilities
// ============================================================================
inline std::string random_id() {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, sizeof(alphanum) - 2);

    std::string result;
    result.reserve(8);
    for (int i = 0; i < 8; ++i) {
        result += alphanum[dis(gen)];
    }
    return result;
}

// Note: print_debug requires pybind11 for Python print capture
// Each backend should provide its own implementation

// ============================================================================
// Core MPI wrappers
// ============================================================================
// These are inline functions that can be used by both CPU and CUDA backends.
// The backends can add their own debug logging around these calls.

inline void mpi_sendrecv_impl(
    void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
    void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
    MPI_Comm comm, MPI_Status* status
) {
    int ierr = MPI_Sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    );
    abort_on_error(ierr, comm, "Sendrecv");
}

inline void mpi_allreduce_impl(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
) {
    int ierr = MPI_Allreduce(sendbuf, recvbuf, count, dtype, op, comm);
    abort_on_error(ierr, comm, "Allreduce");
}

inline void mpi_allgather_impl(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm
) {
    int ierr = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    abort_on_error(ierr, comm, "Allgather");
}

inline void mpi_alltoall_impl(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm
) {
    int ierr = MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    abort_on_error(ierr, comm, "Alltoall");
}

inline void mpi_barrier_impl(MPI_Comm comm) {
    int ierr = MPI_Barrier(comm);
    abort_on_error(ierr, comm, "Barrier");
}

inline void mpi_bcast_impl(void* buf, int count, MPI_Datatype dtype, int root, MPI_Comm comm) {
    int ierr = MPI_Bcast(buf, count, dtype, root, comm);
    abort_on_error(ierr, comm, "Bcast");
}

inline void mpi_gather_impl(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm
) {
    int ierr = MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    abort_on_error(ierr, comm, "Gather");
}

inline void mpi_scatter_impl(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm
) {
    int ierr = MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
    abort_on_error(ierr, comm, "Scatter");
}

inline void mpi_reduce_impl(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, int root, MPI_Comm comm
) {
    int ierr = MPI_Reduce(sendbuf, recvbuf, count, dtype, op, root, comm);
    abort_on_error(ierr, comm, "Reduce");
}

inline void mpi_scan_impl(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
) {
    int ierr = MPI_Scan(sendbuf, recvbuf, count, dtype, op, comm);
    abort_on_error(ierr, comm, "Scan");
}

inline void mpi_send_impl(void* buf, int count, MPI_Datatype dtype, int dest, int tag, MPI_Comm comm) {
    int ierr = MPI_Send(buf, count, dtype, dest, tag, comm);
    abort_on_error(ierr, comm, "Send");
}

inline void mpi_recv_impl(
    void* buf, int count, MPI_Datatype dtype,
    int source, int tag, MPI_Comm comm, MPI_Status* status
) {
    int ierr = MPI_Recv(buf, count, dtype, source, tag, comm, status);
    abort_on_error(ierr, comm, "Recv");
}

} // namespace mpi4jax

#endif // MPI4JAX_MPI_OPS_COMMON_H_
