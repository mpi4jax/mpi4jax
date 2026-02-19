// pybind11 bindings for mpi4jax CPU custom call targets using the new XLA FFI API
// This module provides a C++ alternative to the Cython mpi_xla_bridge_cpu module

#include <pybind11/pybind11.h>
#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <type_traits>

// XLA FFI headers from jaxlib
#include "xla/ffi/api/ffi.h"

// Shared mpi4jax headers
#include "mpi_descriptors.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace mpi4jax {

// ============================================================================
// Global debug flag
// ============================================================================
static bool PRINT_DEBUG = false;

void set_logging(bool enable) {
    PRINT_DEBUG = enable;
}

bool get_logging() {
    return PRINT_DEBUG;
}

// ============================================================================
// Logging utilities
// ============================================================================
static std::string random_id() {
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

static void print_debug(const char* message, const char* rid, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    // Use Python's print to ensure proper pytest capsys capture
    py::gil_scoped_acquire acquire;
    py::print("r" + std::to_string(rank) + " | " + std::string(rid) + " | " + std::string(message), py::arg("flush") = true);
}

// ============================================================================
// Error handling
// ============================================================================
static int abort_on_error(int ierr, MPI_Comm comm, const char* mpi_op) {
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
// MPI_STATUS_IGNORE address
// ============================================================================
uintptr_t get_mpi_status_ignore_addr() {
    return reinterpret_cast<uintptr_t>(MPI_STATUS_IGNORE);
}

// ============================================================================
// Core MPI wrappers with debug logging
// ============================================================================

// --- Barrier ---
static void mpi_barrier(MPI_Comm comm) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        print_debug("MPI_Barrier", rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Barrier(comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Barrier done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Barrier");
}

// --- Allgather ---
static void mpi_allgather(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Allgather (%d items)", sendcount);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Allgather done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Allgather");
}

// --- Allreduce ---
static void mpi_allreduce(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Allreduce with %d items", count);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Allreduce(sendbuf, recvbuf, count, dtype, op, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Allreduce done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Allreduce");
}

// --- Alltoall ---
static void mpi_alltoall(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Alltoall (%d items)", sendcount);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Alltoall done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Alltoall");
}

// --- Bcast ---
static void mpi_bcast(void* buf, int count, MPI_Datatype dtype, int root, MPI_Comm comm) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Bcast (%d items, root=%d)", count, root);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Bcast(buf, count, dtype, root, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Bcast done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Bcast");
}

// --- Gather ---
static void mpi_gather(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Gather (%d items, root=%d)", sendcount, root);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Gather done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Gather");
}

// --- Scatter ---
static void mpi_scatter(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Scatter (%d items, root=%d)", sendcount, root);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Scatter done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Scatter");
}

// --- Reduce ---
static void mpi_reduce(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, int root, MPI_Comm comm
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Reduce (%d items, root=%d)", count, root);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Reduce(sendbuf, recvbuf, count, dtype, op, root, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Reduce done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Reduce");
}

// --- Scan ---
static void mpi_scan(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Scan (%d items)", count);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Scan(sendbuf, recvbuf, count, dtype, op, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Scan done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Scan");
}

// --- Send ---
static void mpi_send(void* buf, int count, MPI_Datatype dtype, int dest, int tag, MPI_Comm comm) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Send -> %d (tag %d, %d items)", dest, tag, count);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Send(buf, count, dtype, dest, tag, comm);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Send done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Send");
}

// --- Recv ---
static void mpi_recv(
    void* buf, int count, MPI_Datatype dtype,
    int source, int tag, MPI_Comm comm, MPI_Status* status
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Recv <- %d (tag %d, %d items)", source, tag, count);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Recv(buf, count, dtype, source, tag, comm, status);

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Recv done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Recv");
}

// --- Sendrecv ---
static void mpi_sendrecv(
    void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
    void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
    MPI_Comm comm, MPI_Status* status
) {
    std::string rid;
    std::chrono::high_resolution_clock::time_point start;

    if (PRINT_DEBUG) {
        rid = random_id();
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "MPI_Sendrecv <- %d (tag %d, %d items) / -> %d (tag %d, %d items)",
                 source, recvtag, recvcount, dest, sendtag, sendcount);
        print_debug(msg, rid.c_str(), comm);
        start = std::chrono::high_resolution_clock::now();
    }

    int ierr = MPI_Sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    );

    if (PRINT_DEBUG) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1e6;
        char msg[256];
        snprintf(msg, sizeof(msg), "MPI_Sendrecv done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Sendrecv");
}

// ============================================================================
// XLA FFI handlers (new typed API)
// ============================================================================

// --- Barrier FFI ---
ffi::Error mpi_barrier_ffi_impl(
    ffi::Token token_in,
    ffi::Result<ffi::Token> token_out,
    int64_t comm_handle
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    mpi_barrier(comm);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_barrier_ffi,
    mpi_barrier_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("comm")
);

// --- Allgather FFI ---
ffi::Error mpi_allgather_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t sendtype_handle,
    int64_t recvcount,
    int64_t recvtype_handle,
    int64_t comm_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_allgather(send_data, static_cast<int>(sendcount), sendtype,
                  recv_data, static_cast<int>(recvcount), recvtype, comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_allgather_ffi,
    mpi_allgather_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("comm")
);

// --- Allreduce FFI ---
ffi::Error mpi_allreduce_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t nitems,
    int64_t op_handle,
    int64_t comm_handle,
    int64_t dtype_handle
) {
    MPI_Op op = from_handle<MPI_Op>(op_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Datatype dtype = from_handle<MPI_Datatype>(dtype_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_allreduce(send_data, recv_data, static_cast<int>(nitems), dtype, op, comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_allreduce_ffi,
    mpi_allreduce_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("op")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Alltoall FFI ---
ffi::Error mpi_alltoall_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t sendtype_handle,
    int64_t recvcount,
    int64_t recvtype_handle,
    int64_t comm_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_alltoall(send_data, static_cast<int>(sendcount), sendtype,
                 recv_data, static_cast<int>(recvcount), recvtype, comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_alltoall_ffi,
    mpi_alltoall_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("comm")
);

// --- Bcast FFI ---
// Note: bcast is special - the root's sendbuf is used, others use recvbuf
ffi::Error mpi_bcast_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t nitems,
    int64_t root,
    int64_t comm_handle,
    int64_t dtype_handle
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Datatype dtype = from_handle<MPI_Datatype>(dtype_handle);

    int rank;
    int ierr = MPI_Comm_rank(comm, &rank);
    abort_on_error(ierr, comm, "Comm_rank");

    void* buf;
    if (rank == static_cast<int>(root)) {
        buf = sendbuf.untyped_data();
    } else {
        buf = recvbuf->untyped_data();
    }

    mpi_bcast(buf, static_cast<int>(nitems), dtype, static_cast<int>(root), comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_bcast_ffi,
    mpi_bcast_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Gather FFI ---
ffi::Error mpi_gather_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t sendtype_handle,
    int64_t recvcount,
    int64_t recvtype_handle,
    int64_t root,
    int64_t comm_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_gather(send_data, static_cast<int>(sendcount), sendtype,
               recv_data, static_cast<int>(recvcount), recvtype,
               static_cast<int>(root), comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_gather_ffi,
    mpi_gather_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
);

// --- Scatter FFI ---
ffi::Error mpi_scatter_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t sendtype_handle,
    int64_t recvcount,
    int64_t recvtype_handle,
    int64_t root,
    int64_t comm_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_scatter(send_data, static_cast<int>(sendcount), sendtype,
                recv_data, static_cast<int>(recvcount), recvtype,
                static_cast<int>(root), comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_scatter_ffi,
    mpi_scatter_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
);

// --- Reduce FFI ---
ffi::Error mpi_reduce_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t nitems,
    int64_t op_handle,
    int64_t root,
    int64_t comm_handle,
    int64_t dtype_handle
) {
    MPI_Op op = from_handle<MPI_Op>(op_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Datatype dtype = from_handle<MPI_Datatype>(dtype_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_reduce(send_data, recv_data, static_cast<int>(nitems), dtype, op,
               static_cast<int>(root), comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_reduce_ffi,
    mpi_reduce_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("op")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Scan FFI ---
ffi::Error mpi_scan_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t nitems,
    int64_t op_handle,
    int64_t comm_handle,
    int64_t dtype_handle
) {
    MPI_Op op = from_handle<MPI_Op>(op_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Datatype dtype = from_handle<MPI_Datatype>(dtype_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_scan(send_data, recv_data, static_cast<int>(nitems), dtype, op, comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_scan_ffi,
    mpi_scan_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("op")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Send FFI ---
// Note: send has no output buffer, but we need token for ordering
ffi::Error mpi_send_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::Token> token_out,
    int64_t nitems,
    int64_t dest,
    int64_t tag,
    int64_t comm_handle,
    int64_t dtype_handle
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Datatype dtype = from_handle<MPI_Datatype>(dtype_handle);

    void* send_data = sendbuf.untyped_data();

    mpi_send(send_data, static_cast<int>(nitems), dtype,
             static_cast<int>(dest), static_cast<int>(tag), comm);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_send_ffi,
    mpi_send_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("dest")
        .Attr<int64_t>("tag")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Recv FFI ---
ffi::Error mpi_recv_ffi_impl(
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t nitems,
    int64_t source,
    int64_t tag,
    int64_t comm_handle,
    int64_t dtype_handle,
    int64_t status_ptr
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Datatype dtype = from_handle<MPI_Datatype>(dtype_handle);
    MPI_Status* status = from_handle<MPI_Status*>(status_ptr);

    void* recv_data = recvbuf->untyped_data();

    mpi_recv(recv_data, static_cast<int>(nitems), dtype,
             static_cast<int>(source), static_cast<int>(tag), comm, status);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_recv_ffi,
    mpi_recv_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("source")
        .Attr<int64_t>("tag")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
        .Attr<int64_t>("status")
);

// --- Sendrecv FFI ---
ffi::Error mpi_sendrecv_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t dest,
    int64_t sendtag,
    int64_t sendtype_handle,
    int64_t recvcount,
    int64_t source,
    int64_t recvtag,
    int64_t recvtype_handle,
    int64_t comm_handle,
    int64_t status_ptr
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Status* status = from_handle<MPI_Status*>(status_ptr);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_sendrecv(
        send_data, static_cast<int>(sendcount), sendtype, static_cast<int>(dest), static_cast<int>(sendtag),
        recv_data, static_cast<int>(recvcount), recvtype, static_cast<int>(source), static_cast<int>(recvtag),
        comm, status
    );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_sendrecv_ffi,
    mpi_sendrecv_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("dest")
        .Attr<int64_t>("sendtag")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("source")
        .Attr<int64_t>("recvtag")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("status")
);

// ============================================================================
// Descriptor-based FFI handlers
// These handlers receive parameters via a descriptor buffer instead of
// individual attributes. This enables code sharing between CPU and CUDA.
// ============================================================================

// --- Sendrecv Descriptor FFI ---
// Receives a SendrecvDescriptor buffer containing all parameters
ffi::Error mpi_sendrecv_desc_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::AnyBuffer descriptor,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out
) {
    // Validate descriptor size
    auto desc_dims = descriptor.dimensions();
    size_t desc_size = 1;
    for (auto d : desc_dims) {
        desc_size *= d;
    }
    if (desc_size != sizeof(SendrecvDescriptor)) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "SendrecvDescriptor size mismatch: expected " +
            std::to_string(sizeof(SendrecvDescriptor)) +
            " bytes, got " + std::to_string(desc_size));
    }

    const SendrecvDescriptor* desc =
        static_cast<const SendrecvDescriptor*>(descriptor.untyped_data());

    MPI_Datatype sendtype = from_handle<MPI_Datatype>(desc->sendtype);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(desc->recvtype);
    MPI_Comm comm = from_handle<MPI_Comm>(desc->comm);
    MPI_Status* status = from_handle<MPI_Status*>(desc->status);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_sendrecv(
        send_data, static_cast<int>(desc->sendcount), sendtype,
        static_cast<int>(desc->dest), static_cast<int>(desc->sendtag),
        recv_data, static_cast<int>(desc->recvcount), recvtype,
        static_cast<int>(desc->source), static_cast<int>(desc->recvtag),
        comm, status
    );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_sendrecv_desc_ffi,
    mpi_sendrecv_desc_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Arg<ffi::AnyBuffer>()  // descriptor
        .Arg<ffi::Token>()      // token_in for ordering
        .Ret<ffi::AnyBuffer>()  // recvbuf
        .Ret<ffi::Token>()      // token_out for ordering
);

// ============================================================================
// Helper to create PyCapsules for XLA custom call registration
// ============================================================================
static py::capsule make_ffi_capsule(void* fn) {
    return py::capsule(fn, "xla._CUSTOM_CALL_TARGET");
}

// ============================================================================
// pybind11 module definition
// ============================================================================
PYBIND11_MODULE(mpi_xla_bridge_cpu, m) {
    m.doc() = "C++ MPI XLA bridge for CPU (pybind11) with XLA FFI support";

    // Expose MPI_STATUS_IGNORE address
    m.attr("MPI_STATUS_IGNORE_ADDR") = get_mpi_status_ignore_addr();

    // Logging control
    m.def("set_logging", &set_logging, "Enable or disable debug logging",
          py::arg("enable"));
    m.def("get_logging", &get_logging, "Get current logging state");

    // Build ffi_targets dictionary for FFI API
    py::dict ffi_targets;
    ffi_targets["mpi_barrier"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_barrier_ffi));
    ffi_targets["mpi_allgather"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_allgather_ffi));
    ffi_targets["mpi_allreduce"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_allreduce_ffi));
    ffi_targets["mpi_alltoall"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_alltoall_ffi));
    ffi_targets["mpi_bcast"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_bcast_ffi));
    ffi_targets["mpi_gather"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_gather_ffi));
    ffi_targets["mpi_scatter"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_scatter_ffi));
    ffi_targets["mpi_reduce"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_reduce_ffi));
    ffi_targets["mpi_scan"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_scan_ffi));
    ffi_targets["mpi_send"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_send_ffi));
    ffi_targets["mpi_recv"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_recv_ffi));
    ffi_targets["mpi_sendrecv"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_sendrecv_ffi));
    // Descriptor-based FFI targets (for unified CPU/CUDA code path)
    ffi_targets["mpi_sendrecv_desc"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_sendrecv_desc_ffi));
    m.attr("ffi_targets") = ffi_targets;
}

} // namespace mpi4jax
