#ifndef MPI_XLA_BRIDGE_CPP_HPP
#define MPI_XLA_BRIDGE_CPP_HPP

#include <mpi.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <string>

namespace mpi4jax {

// Global debug flag
extern bool PRINT_DEBUG;

// Logging functions
void set_logging(bool enable);
bool get_logging();

// Generate random ID for debug logging
std::string random_id();

// Print debug message with rank info
void print_debug(const char* message, const char* rid, MPI_Comm comm);

// Error handling
int abort_on_error(int ierr, MPI_Comm comm, const char* mpi_op);

// MPI_STATUS_IGNORE address (exported for Python)
uintptr_t get_mpi_status_ignore_addr();

// Core MPI wrapper functions
void mpi_sendrecv(
    void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
    void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
    MPI_Comm comm, MPI_Status* status
);

// CPU XLA custom call target for sendrecv
// Signature: void(void** out_ptr, void** data_ptr)
void mpi_sendrecv_cpu(void** out_ptr, void** data_ptr);

} // namespace mpi4jax

#endif // MPI_XLA_BRIDGE_CPP_HPP
