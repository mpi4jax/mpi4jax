// Common MPI operations and utilities shared between CPU, CUDA, and XPU backends
// This header provides:
// - MPI handle conversion utilities (from_handle/to_handle)
// - Error handling utilities
// - Debug logging utilities (with device tag parameter)
// - Core MPI wrapper functions with debug logging

#ifndef MPI4JAX_MPI_OPS_COMMON_H_
#define MPI4JAX_MPI_OPS_COMMON_H_

#include <nanobind/nanobind.h>
#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <chrono>
#include <random>
#include <string>

// XLA FFI headers from jaxlib
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

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
// MPI_STATUS_IGNORE address
// ============================================================================
inline uintptr_t get_mpi_status_ignore_addr() {
    return reinterpret_cast<uintptr_t>(MPI_STATUS_IGNORE);
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

// Checked malloc for copy-to-host mode (used by CUDA and XPU backends)
inline void* checked_malloc(size_t count, MPI_Comm comm) {
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

// ============================================================================
// Global debug logging flag (shared across all backends)
// ============================================================================
// Only one backend is loaded per process, so a single flag suffices
inline bool& get_print_debug_flag() {
    static bool PRINT_DEBUG = false;
    return PRINT_DEBUG;
}

inline void set_logging(bool enable) {
    get_print_debug_flag() = enable;
}

inline bool get_logging() {
    return get_print_debug_flag();
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

inline void print_debug(const char* message, const char* rid, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    // Use Python's print to ensure proper pytest capsys capture
    nb::gil_scoped_acquire acquire;
    std::string msg = "r" + std::to_string(rank) + " | " + std::string(rid) + " | " + std::string(message);
    nb::print(msg.c_str());
    // Flush stdout
    nb::module_::import_("sys").attr("stdout").attr("flush")();
}

// ============================================================================
// Debug logging helper - wraps MPI calls with timing and logging
// ============================================================================
// This class provides RAII-style debug logging for MPI operations.
// Usage:
//   DebugTimer timer(PRINT_DEBUG, comm, "MPI_Allreduce", "with 100 items", "GPU");
//   // ... do MPI call ...
//   timer.finish(ierr);  // or let destructor handle it

class DebugTimer {
public:
    DebugTimer(bool enabled, MPI_Comm comm, const char* op_name,
               const std::string& details = "", const char* device_tag = "")
        : enabled_(enabled), comm_(comm), op_name_(op_name), device_tag_(device_tag) {
        if (enabled_) {
            rid_ = random_id();
            std::string msg = op_name_;
            if (device_tag_[0] != '\0') {
                msg += " (";
                msg += device_tag_;
                msg += ")";
            }
            if (!details.empty()) {
                msg += " ";
                msg += details;
            }
            print_debug(msg.c_str(), rid_.c_str(), comm_);
            start_ = std::chrono::high_resolution_clock::now();
        }
    }

    void finish(int ierr = 0) {
        if (enabled_ && !finished_) {
            finished_ = true;
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            double seconds = duration.count() / 1e6;
            char msg[256];
            std::string op_with_tag = op_name_;
            if (device_tag_[0] != '\0') {
                op_with_tag += " (";
                op_with_tag += device_tag_;
                op_with_tag += ")";
            }
            snprintf(msg, sizeof(msg), "%s done with code %d (%.2es)", op_with_tag.c_str(), ierr, seconds);
            print_debug(msg, rid_.c_str(), comm_);
        }
    }

    ~DebugTimer() {
        finish(0);
    }

private:
    bool enabled_;
    bool finished_ = false;
    MPI_Comm comm_;
    const char* op_name_;
    const char* device_tag_;
    std::string rid_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Core MPI wrappers with debug logging
// ============================================================================
// These functions call get_print_debug_flag() internally for debug logging.
// The device_tag parameter is used for platform-specific logging (e.g., "GPU", "XPU").

inline void mpi_barrier(MPI_Comm comm, const char* device_tag = "") {
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Barrier", "", device_tag);
    int ierr = MPI_Barrier(comm);
    timer.finish(ierr);
    abort_on_error(ierr, comm, "Barrier");
}

inline void mpi_allgather(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm, const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "(%d items)", sendcount);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Allgather", details, device_tag);

    int ierr = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Allgather");
}

inline void mpi_allreduce(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm,
    const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "with %d items", count);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Allreduce", details, device_tag);

    int ierr = MPI_Allreduce(sendbuf, recvbuf, count, dtype, op, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Allreduce");
}

inline void mpi_alltoall(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    MPI_Comm comm, const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "(%d items)", sendcount);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Alltoall", details, device_tag);

    int ierr = MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Alltoall");
}

inline void mpi_bcast(
    void* buf, int count, MPI_Datatype dtype, int root, MPI_Comm comm,
    const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "(%d items, root=%d)", count, root);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Bcast", details, device_tag);

    int ierr = MPI_Bcast(buf, count, dtype, root, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Bcast");
}

inline void mpi_gather(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm, const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "(%d items, root=%d)", sendcount, root);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Gather", details, device_tag);

    int ierr = MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Gather");
}

inline void mpi_scatter(
    void* sendbuf, int sendcount, MPI_Datatype sendtype,
    void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root, MPI_Comm comm, const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "(%d items, root=%d)", sendcount, root);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Scatter", details, device_tag);

    int ierr = MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Scatter");
}

inline void mpi_reduce(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, int root, MPI_Comm comm,
    const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "(%d items, root=%d)", count, root);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Reduce", details, device_tag);

    int ierr = MPI_Reduce(sendbuf, recvbuf, count, dtype, op, root, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Reduce");
}

inline void mpi_scan(
    void* sendbuf, void* recvbuf, int count,
    MPI_Datatype dtype, MPI_Op op, MPI_Comm comm,
    const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "(%d items)", count);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Scan", details, device_tag);

    int ierr = MPI_Scan(sendbuf, recvbuf, count, dtype, op, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Scan");
}

inline void mpi_send(
    void* buf, int count, MPI_Datatype dtype, int dest, int tag, MPI_Comm comm,
    const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "-> %d (tag %d, %d items)", dest, tag, count);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Send", details, device_tag);

    int ierr = MPI_Send(buf, count, dtype, dest, tag, comm);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Send");
}

inline void mpi_recv(
    void* buf, int count, MPI_Datatype dtype,
    int source, int tag, MPI_Comm comm, MPI_Status* status,
    const char* device_tag = ""
) {
    char details[64];
    snprintf(details, sizeof(details), "<- %d (tag %d, %d items)", source, tag, count);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Recv", details, device_tag);

    int ierr = MPI_Recv(buf, count, dtype, source, tag, comm, status);

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Recv");
}

inline void mpi_sendrecv(
    void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
    void* recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
    MPI_Comm comm, MPI_Status* status,
    const char* device_tag = ""
) {
    char details[128];
    snprintf(details, sizeof(details),
             "<- %d (tag %d, %d items) / -> %d (tag %d, %d items)",
             source, recvtag, recvcount, dest, sendtag, sendcount);
    DebugTimer timer(get_print_debug_flag(), comm, "MPI_Sendrecv", details, device_tag);

    int ierr = MPI_Sendrecv(
        sendbuf, sendcount, sendtype, dest, sendtag,
        recvbuf, recvcount, recvtype, source, recvtag,
        comm, status
    );

    timer.finish(ierr);
    abort_on_error(ierr, comm, "Sendrecv");
}

// ============================================================================
// MPI ABI information for compatibility checking
// ============================================================================
// This struct captures build-time information about the MPI implementation
// to detect incompatibilities at runtime (e.g., built with OpenMPI but
// running with MPICH, which have different handle types).

struct MpiAbiInfo {
    size_t sizeof_comm;      // sizeof(MPI_Comm)
    size_t sizeof_datatype;  // sizeof(MPI_Datatype)
    size_t sizeof_op;        // sizeof(MPI_Op)
    size_t sizeof_status;    // sizeof(MPI_Status)
    int64_t comm_world_handle;  // MPI_COMM_WORLD as int64
    const char* mpi_library_version;  // MPI_Get_library_version result
};

inline MpiAbiInfo get_mpi_abi_info() {
    static char lib_version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
    static bool initialized = false;

    if (!initialized) {
        int len;
        MPI_Get_library_version(lib_version, &len);
        initialized = true;
    }

    return MpiAbiInfo{
        sizeof(MPI_Comm),
        sizeof(MPI_Datatype),
        sizeof(MPI_Op),
        sizeof(MPI_Status),
        to_handle(MPI_COMM_WORLD),
        lib_version
    };
}

// ============================================================================
// Nanobind helper for registering logging functions
// ============================================================================
// Used by all backends (CPU, CUDA, XPU)
#define MPI4JAX_REGISTER_LOGGING(m) \
    m.def("set_logging", &set_logging, "Enable or disable debug logging", nb::arg("enable")); \
    m.def("get_logging", &get_logging, "Get current debug logging state")

// ============================================================================
// Helper to encapsulate FFI handler as PyCapsule with compile-time validation
// ============================================================================
// Used by all backends to register FFI targets.
// The static_assert ensures that the handler has the correct signature
// expected by XLA FFI: XLA_FFI_Error* (*)(XLA_FFI_CallFrame*)
template<typename T>
inline nb::capsule encapsulate_ffi_handler(T* fn) {
    // Validate at compile time that fn has the correct XLA FFI handler signature
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
        "Encapsulated function must be an XLA FFI handler with signature: "
        "XLA_FFI_Error* (*)(XLA_FFI_CallFrame*)"
    );
    return nb::capsule(
        reinterpret_cast<void*>(fn),
        "xla._CUSTOM_CALL_TARGET"
    );
}

} // namespace mpi4jax

#endif // MPI4JAX_MPI_OPS_COMMON_H_
