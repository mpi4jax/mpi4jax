// pybind11 bindings for mpi4jax CUDA custom call targets using the XLA FFI API
// This module provides C++ FFI-based GPU primitives, starting with sendrecv

#include <pybind11/pybind11.h>
#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <cstdlib>
#include <type_traits>
#include <cstring>

// CUDA headers
#include <cuda_runtime.h>

// XLA FFI headers from jaxlib
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace mpi4jax {

// ============================================================================
// Global configuration
// ============================================================================
static bool PRINT_DEBUG = false;
static bool COPY_TO_HOST = false;

void set_logging(bool enable) {
    PRINT_DEBUG = enable;
}

bool get_logging() {
    return PRINT_DEBUG;
}

void set_copy_to_host(bool enable) {
    COPY_TO_HOST = enable;
}

bool get_copy_to_host() {
    return COPY_TO_HOST;
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

static void abort_on_cuda_error(cudaError_t ierr, MPI_Comm comm, const char* cuda_op) {
    if (ierr == cudaSuccess) {
        return;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    std::cerr << "r" << rank << " | " << cuda_op
              << " failed with error " << ierr << ": "
              << cudaGetErrorName(ierr) << " - "
              << cudaGetErrorString(ierr) << " - aborting" << std::endl;
    std::cerr.flush();

    MPI_Abort(comm, static_cast<int>(ierr));
}

// ============================================================================
// CUDA memory management utilities
// ============================================================================
static void* checked_malloc(size_t count, MPI_Comm comm) {
    void* mem = std::malloc(count);
    if (!mem) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        std::cerr << "r" << rank << " | Failed to allocate " << count
                  << " bytes on host - aborting" << std::endl;
        std::cerr.flush();
        MPI_Abort(comm, 1);
    }
    return mem;
}

static void checked_cuda_memcpy(void* dst, const void* src, size_t count,
                                 cudaMemcpyKind kind, cudaStream_t stream, MPI_Comm comm) {
    cudaError_t ierr = cudaMemcpyAsync(dst, src, count, kind, stream);
    abort_on_cuda_error(ierr, comm, "cudaMemcpyAsync");

    ierr = cudaStreamSynchronize(stream);
    abort_on_cuda_error(ierr, comm, "cudaStreamSynchronize");
}

static void checked_cuda_stream_synchronize(cudaStream_t stream, MPI_Comm comm) {
    cudaError_t ierr = cudaStreamSynchronize(stream);
    abort_on_cuda_error(ierr, comm, "cudaStreamSynchronize");
}

// ============================================================================
// MPI handle conversion utilities
// ============================================================================
// MPI implementations differ in how they define handle types:
// - OpenMPI: MPI_Comm, MPI_Datatype, MPI_Op are pointers (e.g., ompi_communicator_t*)
// - MPICH: MPI_Comm, MPI_Datatype, MPI_Op are integers (int)
// We use memcpy for type-safe conversion that works with both implementations.

template<typename MPI_Handle>
inline MPI_Handle from_handle(uint64_t handle) {
    MPI_Handle result;
    std::memcpy(&result, &handle, sizeof(result));
    return result;
}

template<typename MPI_Handle>
inline uint64_t to_handle(MPI_Handle handle) {
    uint64_t result = 0;
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
// Core MPI wrappers with debug logging (GPU versions)
// ============================================================================

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
                 "MPI_Sendrecv (GPU) <- %d (tag %d, %d items) / -> %d (tag %d, %d items)",
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
        snprintf(msg, sizeof(msg), "MPI_Sendrecv (GPU) done with code %d (%.2es)", ierr, seconds);
        print_debug(msg, rid.c_str(), comm);
    }

    abort_on_error(ierr, comm, "Sendrecv");
}

// ============================================================================
// XLA FFI handlers for GPU
// ============================================================================

// --- Sendrecv FFI (GPU) ---
ffi::Error mpi_sendrecv_ffi_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t dest,
    int64_t sendtag,
    uint64_t sendtype_handle,
    int64_t recvcount,
    int64_t source,
    int64_t recvtag,
    uint64_t recvtype_handle,
    uint64_t comm_handle,
    uint64_t status_ptr
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Status* status = from_handle<MPI_Status*>(status_ptr);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    // Get the CUDA stream from the execution context
    // For now, we use the default stream (0) and synchronize
    // In the future, this could be obtained from the FFI context
    cudaStream_t stream = 0;

    if (COPY_TO_HOST) {
        // Copy device memory to host, perform MPI operation, copy back
        int send_dtype_size, recv_dtype_size;
        int ierr = MPI_Type_size(sendtype, &send_dtype_size);
        abort_on_error(ierr, comm, "Type_size");

        ierr = MPI_Type_size(recvtype, &recv_dtype_size);
        abort_on_error(ierr, comm, "Type_size");

        size_t bytes_send = static_cast<size_t>(send_dtype_size) * static_cast<size_t>(sendcount);
        size_t bytes_recv = static_cast<size_t>(recv_dtype_size) * static_cast<size_t>(recvcount);

        void* host_sendbuf = checked_malloc(bytes_send, comm);
        void* host_recvbuf = checked_malloc(bytes_recv, comm);

        // Copy send data from device to host
        checked_cuda_memcpy(host_sendbuf, send_data, bytes_send,
                           cudaMemcpyDeviceToHost, stream, comm);

        // Perform MPI sendrecv on host memory
        mpi_sendrecv(
            host_sendbuf, static_cast<int>(sendcount), sendtype, static_cast<int>(dest), static_cast<int>(sendtag),
            host_recvbuf, static_cast<int>(recvcount), recvtype, static_cast<int>(source), static_cast<int>(recvtag),
            comm, status
        );

        // Copy received data from host to device
        checked_cuda_memcpy(recv_data, host_recvbuf, bytes_recv,
                           cudaMemcpyHostToDevice, stream, comm);

        std::free(host_sendbuf);
        std::free(host_recvbuf);
    } else {
        // GPU-aware MPI: operate directly on device memory
        // Just need to synchronize the stream before calling MPI
        checked_cuda_stream_synchronize(stream, comm);

        mpi_sendrecv(
            send_data, static_cast<int>(sendcount), sendtype, static_cast<int>(dest), static_cast<int>(sendtag),
            recv_data, static_cast<int>(recvcount), recvtype, static_cast<int>(source), static_cast<int>(recvtag),
            comm, status
        );
    }

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
        .Attr<uint64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("source")
        .Attr<int64_t>("recvtag")
        .Attr<uint64_t>("recvtype")
        .Attr<uint64_t>("comm")
        .Attr<uint64_t>("status")
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
PYBIND11_MODULE(mpi_xla_bridge_cuda_cpp, m) {
    m.doc() = "C++ MPI XLA bridge for CUDA (pybind11) with XLA FFI support";

    // Expose MPI_STATUS_IGNORE address
    m.attr("MPI_STATUS_IGNORE_ADDR") = get_mpi_status_ignore_addr();

    // Logging control
    m.def("set_logging", &set_logging, "Enable or disable debug logging",
          py::arg("enable"));
    m.def("get_logging", &get_logging, "Get current logging state");

    // Copy-to-host mode control
    m.def("set_copy_to_host", &set_copy_to_host,
          "Enable or disable copy-to-host mode for non-GPU-aware MPI",
          py::arg("enable"));
    m.def("get_copy_to_host", &get_copy_to_host, "Get current copy-to-host state");

    // Build ffi_targets dictionary for FFI API
    py::dict ffi_targets;
    ffi_targets["mpi_sendrecv"] = make_ffi_capsule(reinterpret_cast<void*>(&mpi_sendrecv_ffi));
    m.attr("ffi_targets") = ffi_targets;
}

} // namespace mpi4jax
