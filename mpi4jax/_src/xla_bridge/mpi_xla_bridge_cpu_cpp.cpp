// pybind11 bindings for mpi4jax CPU custom call targets using the new XLA FFI API
// This module provides a C++ alternative to the Cython mpi_xla_bridge_cpu module

#include <pybind11/pybind11.h>
#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <random>
#include <string>

// XLA FFI headers from jaxlib
#include "xla/ffi/api/ffi.h"

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
    std::cout << "r" << rank << " | " << rid << " | " << message << std::endl;
    std::cout.flush();
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
// MPI_STATUS_IGNORE address
// ============================================================================
uintptr_t get_mpi_status_ignore_addr() {
    return reinterpret_cast<uintptr_t>(MPI_STATUS_IGNORE);
}

// ============================================================================
// Core MPI wrapper: sendrecv
// ============================================================================
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

    // MPI Call
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
// XLA FFI handler for sendrecv (new typed API)
//
// This uses ffi::AnyBuffer for input/output to handle any dtype.
// Parameters are passed as attributes.
// ============================================================================
ffi::Error mpi_sendrecv_ffi_impl(
    // Input buffer (send data)
    ffi::AnyBuffer sendbuf,
    // Output buffer (receive data)
    ffi::Result<ffi::AnyBuffer> recvbuf,
    // Attributes (passed as keyword arguments from Python)
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
    // Extract MPI types from handles
    MPI_Datatype sendtype = reinterpret_cast<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = reinterpret_cast<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = reinterpret_cast<MPI_Comm>(comm_handle);
    MPI_Status* status = reinterpret_cast<MPI_Status*>(status_ptr);

    // Get raw data pointers
    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    // Call the MPI wrapper
    mpi_sendrecv(
        send_data, static_cast<int>(sendcount), sendtype, static_cast<int>(dest), static_cast<int>(sendtag),
        recv_data, static_cast<int>(recvcount), recvtype, static_cast<int>(source), static_cast<int>(recvtag),
        comm, status
    );

    return ffi::Error::Success();
}

// Define the FFI handler symbol with the binding specification
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_sendrecv_ffi,
    mpi_sendrecv_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // sendbuf
        .Ret<ffi::AnyBuffer>()  // recvbuf
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
// Legacy API (api_version=0) for backward compatibility
// CPU XLA custom call target for sendrecv
// Signature: void(void** out_ptr, void** data_ptr)
// ============================================================================
static void mpi_sendrecv_cpu_legacy(void** out_ptr, void** data_ptr) {
    // Extract parameters from data_ptr array (same layout as Cython version)
    int sendcount = *reinterpret_cast<int*>(data_ptr[0]);
    void* sendbuf = data_ptr[1];
    int dest = *reinterpret_cast<int*>(data_ptr[2]);
    int sendtag = *reinterpret_cast<int*>(data_ptr[3]);
    MPI_Datatype sendtype = reinterpret_cast<MPI_Datatype>(*reinterpret_cast<uintptr_t*>(data_ptr[4]));

    int recvcount = *reinterpret_cast<int*>(data_ptr[5]);
    int source = *reinterpret_cast<int*>(data_ptr[6]);
    int recvtag = *reinterpret_cast<int*>(data_ptr[7]);
    MPI_Datatype recvtype = reinterpret_cast<MPI_Datatype>(*reinterpret_cast<uintptr_t*>(data_ptr[8]));

    MPI_Comm comm = reinterpret_cast<MPI_Comm>(*reinterpret_cast<uintptr_t*>(data_ptr[9]));
    MPI_Status* status = reinterpret_cast<MPI_Status*>(*reinterpret_cast<uintptr_t*>(data_ptr[10]));

    void* recvbuf = out_ptr[0];

    mpi_sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    );
}

// ============================================================================
// Helper to create PyCapsules for XLA custom call registration
// ============================================================================
static py::capsule make_legacy_capsule(void* fn) {
    // The name "xla._CUSTOM_CALL_TARGET" is required by XLA/JAX for legacy API
    return py::capsule(fn, "xla._CUSTOM_CALL_TARGET");
}

static py::capsule make_ffi_capsule(void* fn) {
    // The name "xla._CUSTOM_CALL_TARGET" is also used for FFI API
    return py::capsule(fn, "xla._CUSTOM_CALL_TARGET");
}

// ============================================================================
// pybind11 module definition
// ============================================================================
PYBIND11_MODULE(mpi_xla_bridge_cpu_cpp, m) {
    m.doc() = "C++ MPI XLA bridge for CPU (pybind11) with XLA FFI support";

    // Expose MPI_STATUS_IGNORE address
    m.attr("MPI_STATUS_IGNORE_ADDR") = get_mpi_status_ignore_addr();

    // Logging control
    m.def("set_logging", &set_logging, "Enable or disable debug logging",
          py::arg("enable"));
    m.def("get_logging", &get_logging, "Get current logging state");

    // Build custom_call_targets dictionary for legacy API (api_version=0)
    // This mirrors the Cython module's custom_call_targets dict
    py::dict custom_call_targets;
    custom_call_targets["mpi_sendrecv"] = make_legacy_capsule(
        reinterpret_cast<void*>(&mpi_sendrecv_cpu_legacy)
    );
    m.attr("custom_call_targets") = custom_call_targets;

    // Build ffi_targets dictionary for new FFI API (api_version=1)
    py::dict ffi_targets;
    ffi_targets["mpi_sendrecv"] = make_ffi_capsule(
        reinterpret_cast<void*>(&mpi_sendrecv_ffi)
    );
    m.attr("ffi_targets") = ffi_targets;
}

} // namespace mpi4jax
