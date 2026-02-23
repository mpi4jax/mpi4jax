// nanobind bindings for mpi4jax XPU (SYCL) custom call targets using the XLA FFI API
// This module provides C++ FFI-based GPU primitives for all MPI operations on Intel XPU
//
// Uses shared code from mpi_ops_common.h for MPI wrappers and utilities.
// XPU-specific code handles copy-to-host mode (always enabled as Intel GPU-aware MPI is less common).

#include <sycl/sycl.hpp>

// Shared mpi4jax header with MPI wrappers and utilities
#include "mpi_ops_common.h"

namespace ffi = xla::ffi;

namespace mpi4jax {

// ============================================================================
// Global configuration
// ============================================================================
static bool COPY_TO_HOST = true;  // XPU always uses copy-to-host by default

void set_copy_to_host(bool enable) {
    COPY_TO_HOST = enable;
}

bool get_copy_to_host() {
    return COPY_TO_HOST;
}

// Device tag for debug logging
constexpr char XPU_DEVICE_TAG[] = "XPU";

// Note: For XPU, we use simple memcpy since the FFI buffers are accessible
// In a full implementation with device memory, we would use sycl::queue::memcpy
// checked_malloc is provided by mpi_ops_common.h

// ============================================================================
// XPU-specific FFI handlers
// For XPU we always use copy-to-host as GPU-aware MPI is less common
// These call the shared MPI wrappers from mpi_ops_common.h
// ============================================================================

// --- Barrier FFI (XPU) ---
ffi::Error mpi_barrier_ffi_xpu_impl(
    ffi::Token token_in,
    ffi::Result<ffi::Token> token_out,
    int64_t comm_handle
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    mpi_barrier(comm, XPU_DEVICE_TAG);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_barrier_ffi,
    mpi_barrier_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Token>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("comm")
);

// --- Allgather FFI (XPU) ---
ffi::Error mpi_allgather_ffi_xpu_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t recvcount,
    int64_t comm_handle,
    int64_t sendtype_handle,
    int64_t recvtype_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    // For XPU we always copy to host
    int sendtype_size, recvtype_size, comm_size;
    MPI_Type_size(sendtype, &sendtype_size);
    MPI_Type_size(recvtype, &recvtype_size);
    MPI_Comm_size(comm, &comm_size);

    size_t sendbytes = static_cast<size_t>(sendtype_size) * sendcount;
    size_t recvbytes = static_cast<size_t>(recvtype_size) * recvcount * comm_size;

    void* host_send = checked_malloc(sendbytes, comm);
    void* host_recv = checked_malloc(recvbytes, comm);

    std::memcpy(host_send, send_data, sendbytes);

    mpi_allgather(host_send, static_cast<int>(sendcount), sendtype,
                  host_recv, static_cast<int>(recvcount), recvtype, comm,
                  XPU_DEVICE_TAG);

    std::memcpy(recv_data, host_recv, recvbytes);

    std::free(host_send);
    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_allgather_ffi,
    mpi_allgather_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvtype")
);

// --- Allreduce FFI (XPU) ---
ffi::Error mpi_allreduce_ffi_xpu_impl(
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

    int dtype_size;
    MPI_Type_size(dtype, &dtype_size);
    size_t count = static_cast<size_t>(dtype_size) * nitems;

    void* host_send = checked_malloc(count, comm);
    void* host_recv = checked_malloc(count, comm);

    std::memcpy(host_send, send_data, count);

    mpi_allreduce(host_send, host_recv, static_cast<int>(nitems), dtype, op, comm,
                  XPU_DEVICE_TAG);

    std::memcpy(recv_data, host_recv, count);

    std::free(host_send);
    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_allreduce_ffi,
    mpi_allreduce_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("op")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Alltoall FFI (XPU) ---
ffi::Error mpi_alltoall_ffi_xpu_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t recvcount,
    int64_t comm_handle,
    int64_t sendtype_handle,
    int64_t recvtype_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    int sendtype_size, recvtype_size, comm_size;
    MPI_Type_size(sendtype, &sendtype_size);
    MPI_Type_size(recvtype, &recvtype_size);
    MPI_Comm_size(comm, &comm_size);

    size_t sendbytes = static_cast<size_t>(sendtype_size) * sendcount * comm_size;
    size_t recvbytes = static_cast<size_t>(recvtype_size) * recvcount * comm_size;

    void* host_send = checked_malloc(sendbytes, comm);
    void* host_recv = checked_malloc(recvbytes, comm);

    std::memcpy(host_send, send_data, sendbytes);

    mpi_alltoall(host_send, static_cast<int>(sendcount), sendtype,
                 host_recv, static_cast<int>(recvcount), recvtype, comm,
                 XPU_DEVICE_TAG);

    std::memcpy(recv_data, host_recv, recvbytes);

    std::free(host_send);
    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_alltoall_ffi,
    mpi_alltoall_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvtype")
);

// --- Bcast FFI (XPU) ---
ffi::Error mpi_bcast_ffi_xpu_impl(
    ffi::AnyBuffer buf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::Token> token_out,
    int64_t nitems,
    int64_t root,
    int64_t comm_handle,
    int64_t dtype_handle
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Datatype dtype = from_handle<MPI_Datatype>(dtype_handle);

    int rank;
    MPI_Comm_rank(comm, &rank);

    void* in_data = buf.untyped_data();
    void* out_data = out->untyped_data();

    int dtype_size;
    MPI_Type_size(dtype, &dtype_size);
    size_t count = static_cast<size_t>(dtype_size) * nitems;

    void* host_buf = checked_malloc(count, comm);

    if (rank == static_cast<int>(root)) {
        std::memcpy(host_buf, in_data, count);
    }

    mpi_bcast(host_buf, static_cast<int>(nitems), dtype, static_cast<int>(root), comm,
              XPU_DEVICE_TAG);

    if (rank != static_cast<int>(root)) {
        std::memcpy(out_data, host_buf, count);
    }

    std::free(host_buf);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_bcast_ffi,
    mpi_bcast_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Gather FFI (XPU) ---
ffi::Error mpi_gather_ffi_xpu_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t recvcount,
    int64_t root,
    int64_t comm_handle,
    int64_t sendtype_handle,
    int64_t recvtype_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);

    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    int sendtype_size, recvtype_size;
    MPI_Type_size(sendtype, &sendtype_size);
    MPI_Type_size(recvtype, &recvtype_size);

    size_t sendbytes = static_cast<size_t>(sendtype_size) * sendcount;
    size_t recvbytes = static_cast<size_t>(recvtype_size) * recvcount;
    if (rank == static_cast<int>(root)) {
        recvbytes *= comm_size;
    }

    void* host_send = checked_malloc(sendbytes, comm);
    void* host_recv = checked_malloc(recvbytes, comm);

    std::memcpy(host_send, send_data, sendbytes);

    mpi_gather(host_send, static_cast<int>(sendcount), sendtype,
               host_recv, static_cast<int>(recvcount), recvtype,
               static_cast<int>(root), comm, XPU_DEVICE_TAG);

    if (rank == static_cast<int>(root)) {
        std::memcpy(recv_data, host_recv, recvbytes);
    }

    std::free(host_send);
    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_gather_ffi,
    mpi_gather_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvtype")
);

// --- Scatter FFI (XPU) ---
ffi::Error mpi_scatter_ffi_xpu_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t recvcount,
    int64_t root,
    int64_t comm_handle,
    int64_t sendtype_handle,
    int64_t recvtype_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);

    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    int sendtype_size, recvtype_size;
    MPI_Type_size(sendtype, &sendtype_size);
    MPI_Type_size(recvtype, &recvtype_size);

    size_t sendbytes = static_cast<size_t>(sendtype_size) * sendcount;
    if (rank == static_cast<int>(root)) {
        sendbytes *= comm_size;
    }
    size_t recvbytes = static_cast<size_t>(recvtype_size) * recvcount;

    void* host_send = checked_malloc(sendbytes, comm);
    void* host_recv = checked_malloc(recvbytes, comm);

    std::memcpy(host_send, send_data, sendbytes);

    mpi_scatter(host_send, static_cast<int>(sendcount), sendtype,
                host_recv, static_cast<int>(recvcount), recvtype,
                static_cast<int>(root), comm, XPU_DEVICE_TAG);

    std::memcpy(recv_data, host_recv, recvbytes);

    std::free(host_send);
    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_scatter_ffi,
    mpi_scatter_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvtype")
);

// --- Reduce FFI (XPU) ---
ffi::Error mpi_reduce_ffi_xpu_impl(
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

    int rank;
    MPI_Comm_rank(comm, &rank);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    int dtype_size;
    MPI_Type_size(dtype, &dtype_size);
    size_t count = static_cast<size_t>(dtype_size) * nitems;

    void* host_send = checked_malloc(count, comm);
    void* host_recv = checked_malloc(count, comm);

    std::memcpy(host_send, send_data, count);

    mpi_reduce(host_send, host_recv, static_cast<int>(nitems), dtype, op,
               static_cast<int>(root), comm, XPU_DEVICE_TAG);

    if (rank == static_cast<int>(root)) {
        std::memcpy(recv_data, host_recv, count);
    }

    std::free(host_send);
    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_reduce_ffi,
    mpi_reduce_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("op")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Scan FFI (XPU) ---
ffi::Error mpi_scan_ffi_xpu_impl(
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

    int dtype_size;
    MPI_Type_size(dtype, &dtype_size);
    size_t count = static_cast<size_t>(dtype_size) * nitems;

    void* host_send = checked_malloc(count, comm);
    void* host_recv = checked_malloc(count, comm);

    std::memcpy(host_send, send_data, count);

    mpi_scan(host_send, host_recv, static_cast<int>(nitems), dtype, op, comm,
             XPU_DEVICE_TAG);

    std::memcpy(recv_data, host_recv, count);

    std::free(host_send);
    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_scan_ffi,
    mpi_scan_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("op")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Send FFI (XPU) ---
ffi::Error mpi_send_ffi_xpu_impl(
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

    int dtype_size;
    MPI_Type_size(dtype, &dtype_size);
    size_t count = static_cast<size_t>(dtype_size) * nitems;

    void* host_send = checked_malloc(count, comm);

    std::memcpy(host_send, send_data, count);

    mpi_send(host_send, static_cast<int>(nitems), dtype,
             static_cast<int>(dest), static_cast<int>(tag), comm,
             XPU_DEVICE_TAG);

    std::free(host_send);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_send_ffi,
    mpi_send_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("dest")
        .Attr<int64_t>("tag")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Recv FFI (XPU) ---
ffi::Error mpi_recv_ffi_xpu_impl(
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t nitems,
    int64_t source,
    int64_t tag,
    int64_t comm_handle,
    int64_t dtype_handle,
    int64_t status_handle
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Datatype dtype = from_handle<MPI_Datatype>(dtype_handle);
    MPI_Status* status = from_handle<MPI_Status*>(status_handle);

    void* recv_data = recvbuf->untyped_data();

    int dtype_size;
    MPI_Type_size(dtype, &dtype_size);
    size_t count = static_cast<size_t>(dtype_size) * nitems;

    void* host_recv = checked_malloc(count, comm);

    mpi_recv(host_recv, static_cast<int>(nitems), dtype,
             static_cast<int>(source), static_cast<int>(tag), comm, status,
             XPU_DEVICE_TAG);

    std::memcpy(recv_data, host_recv, count);

    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_recv_ffi,
    mpi_recv_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("source")
        .Attr<int64_t>("tag")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
        .Attr<int64_t>("status")
);

// --- Sendrecv FFI (XPU) ---
ffi::Error mpi_sendrecv_ffi_xpu_impl(
    ffi::AnyBuffer sendbuf,
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> recvbuf,
    ffi::Result<ffi::Token> token_out,
    int64_t sendcount,
    int64_t dest,
    int64_t sendtag,
    int64_t recvcount,
    int64_t source,
    int64_t recvtag,
    int64_t comm_handle,
    int64_t sendtype_handle,
    int64_t recvtype_handle,
    int64_t status_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Status* status = from_handle<MPI_Status*>(status_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    int send_dtype_size, recv_dtype_size;
    MPI_Type_size(sendtype, &send_dtype_size);
    MPI_Type_size(recvtype, &recv_dtype_size);

    size_t bytes_send = static_cast<size_t>(send_dtype_size) * sendcount;
    size_t bytes_recv = static_cast<size_t>(recv_dtype_size) * recvcount;

    void* host_send = checked_malloc(bytes_send, comm);
    void* host_recv = checked_malloc(bytes_recv, comm);

    std::memcpy(host_send, send_data, bytes_send);

    mpi_sendrecv(
        host_send, static_cast<int>(sendcount), sendtype,
        static_cast<int>(dest), static_cast<int>(sendtag),
        host_recv, static_cast<int>(recvcount), recvtype,
        static_cast<int>(source), static_cast<int>(recvtag),
        comm, status, XPU_DEVICE_TAG
    );

    std::memcpy(recv_data, host_recv, bytes_recv);

    std::free(host_send);
    std::free(host_recv);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_sendrecv_ffi,
    mpi_sendrecv_ffi_xpu_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("dest")
        .Attr<int64_t>("sendtag")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("source")
        .Attr<int64_t>("recvtag")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("status")
);

// ============================================================================
// nanobind module definition
// ============================================================================
NB_MODULE(mpi_xla_bridge_xpu, m) {
    m.doc() = "C++ MPI XLA bridge for XPU/SYCL (nanobind) with XLA FFI support";

    // Logging control (uses shared functions from mpi_ops_common.h)
    MPI4JAX_REGISTER_LOGGING(m);

    // Copy-to-host mode control
    m.def("set_copy_to_host", &set_copy_to_host,
          "Enable or disable copy-to-host mode (always enabled for XPU currently)",
          nb::arg("enable"));
    m.def("get_copy_to_host", &get_copy_to_host, "Get current copy-to-host state");

    // Build ffi_targets dictionary for attribute-based FFI API
    nb::dict ffi_targets;
    ffi_targets["mpi_barrier"] = encapsulate_ffi_handler(mpi_barrier_ffi);
    ffi_targets["mpi_allgather"] = encapsulate_ffi_handler(mpi_allgather_ffi);
    ffi_targets["mpi_allreduce"] = encapsulate_ffi_handler(mpi_allreduce_ffi);
    ffi_targets["mpi_alltoall"] = encapsulate_ffi_handler(mpi_alltoall_ffi);
    ffi_targets["mpi_bcast"] = encapsulate_ffi_handler(mpi_bcast_ffi);
    ffi_targets["mpi_gather"] = encapsulate_ffi_handler(mpi_gather_ffi);
    ffi_targets["mpi_scatter"] = encapsulate_ffi_handler(mpi_scatter_ffi);
    ffi_targets["mpi_reduce"] = encapsulate_ffi_handler(mpi_reduce_ffi);
    ffi_targets["mpi_scan"] = encapsulate_ffi_handler(mpi_scan_ffi);
    ffi_targets["mpi_send"] = encapsulate_ffi_handler(mpi_send_ffi);
    ffi_targets["mpi_recv"] = encapsulate_ffi_handler(mpi_recv_ffi);
    ffi_targets["mpi_sendrecv"] = encapsulate_ffi_handler(mpi_sendrecv_ffi);
    m.attr("ffi_targets") = ffi_targets;
}

} // namespace mpi4jax
