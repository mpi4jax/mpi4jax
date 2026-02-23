// nanobind bindings for mpi4jax CUDA custom call targets using the XLA FFI API
// This module provides C++ FFI-based GPU primitives for all MPI operations
//
// Uses shared code from mpi_ops_common.h for MPI wrappers and utilities.
// CUDA-specific code handles COPY_TO_HOST mode for non-GPU-aware MPI.

#include <cuda_runtime.h>

// Shared mpi4jax header with MPI wrappers and utilities
#include "mpi_ops_common.h"

namespace ffi = xla::ffi;

namespace mpi4jax {

// ============================================================================
// Global configuration
// ============================================================================
static bool COPY_TO_HOST = false;

void set_copy_to_host(bool enable) {
    COPY_TO_HOST = enable;
}

bool get_copy_to_host() {
    return COPY_TO_HOST;
}

// Device tag for debug logging
constexpr char CUDA_DEVICE_TAG[] = "GPU";

// ============================================================================
// CUDA error handling
// ============================================================================
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
// CUDA-specific FFI handlers (with COPY_TO_HOST support)
// These call the shared MPI wrappers from mpi_ops_common.h
// ============================================================================

// --- Barrier FFI (GPU) ---
ffi::Error mpi_barrier_ffi_cuda_impl(
    cudaStream_t stream,
    ffi::Token token_in,
    ffi::Result<ffi::Token> token_out,
    int64_t comm_handle
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    // Synchronize the stream to ensure all previous GPU work is complete
    checked_cuda_stream_synchronize(stream, comm);
    mpi_barrier(comm, CUDA_DEVICE_TAG);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_barrier_ffi,
    mpi_barrier_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Token>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("comm")
);

// --- Allgather FFI (GPU) ---
ffi::Error mpi_allgather_ffi_cuda_impl(
    cudaStream_t stream,
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

    if (COPY_TO_HOST) {
        int sendtype_size, recvtype_size, comm_size;
        MPI_Type_size(sendtype, &sendtype_size);
        MPI_Type_size(recvtype, &recvtype_size);
        MPI_Comm_size(comm, &comm_size);

        size_t sendbytes = static_cast<size_t>(sendtype_size) * sendcount;
        size_t recvbytes = static_cast<size_t>(recvtype_size) * recvcount * comm_size;

        void* host_send = checked_malloc(sendbytes, comm);
        void* host_recv = checked_malloc(recvbytes, comm);

        checked_cuda_memcpy(host_send, send_data, sendbytes, cudaMemcpyDeviceToHost, stream, comm);

        mpi_allgather(host_send, static_cast<int>(sendcount), sendtype,
                      host_recv, static_cast<int>(recvcount), recvtype, comm,
                      CUDA_DEVICE_TAG);

        checked_cuda_memcpy(recv_data, host_recv, recvbytes, cudaMemcpyHostToDevice, stream, comm);

        std::free(host_send);
        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_allgather(send_data, static_cast<int>(sendcount), sendtype,
                      recv_data, static_cast<int>(recvcount), recvtype, comm,
                      CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_allgather_ffi,
    mpi_allgather_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("comm")
);

// --- Allreduce FFI (GPU) ---
ffi::Error mpi_allreduce_ffi_cuda_impl(
    cudaStream_t stream,
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

    if (COPY_TO_HOST) {
        int dtype_size;
        MPI_Type_size(dtype, &dtype_size);
        size_t count = static_cast<size_t>(dtype_size) * nitems;

        void* host_send = checked_malloc(count, comm);
        void* host_recv = checked_malloc(count, comm);

        checked_cuda_memcpy(host_send, send_data, count, cudaMemcpyDeviceToHost, stream, comm);

        mpi_allreduce(host_send, host_recv, static_cast<int>(nitems), dtype, op, comm,
                      CUDA_DEVICE_TAG);

        checked_cuda_memcpy(recv_data, host_recv, count, cudaMemcpyHostToDevice, stream, comm);

        std::free(host_send);
        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_allreduce(send_data, recv_data, static_cast<int>(nitems), dtype, op, comm,
                      CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_allreduce_ffi,
    mpi_allreduce_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("op")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Alltoall FFI (GPU) ---
ffi::Error mpi_alltoall_ffi_cuda_impl(
    cudaStream_t stream,
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

    if (COPY_TO_HOST) {
        int sendtype_size, recvtype_size, comm_size;
        MPI_Type_size(sendtype, &sendtype_size);
        MPI_Type_size(recvtype, &recvtype_size);
        MPI_Comm_size(comm, &comm_size);

        size_t sendbytes = static_cast<size_t>(sendtype_size) * sendcount * comm_size;
        size_t recvbytes = static_cast<size_t>(recvtype_size) * recvcount * comm_size;

        void* host_send = checked_malloc(sendbytes, comm);
        void* host_recv = checked_malloc(recvbytes, comm);

        checked_cuda_memcpy(host_send, send_data, sendbytes, cudaMemcpyDeviceToHost, stream, comm);

        mpi_alltoall(host_send, static_cast<int>(sendcount), sendtype,
                     host_recv, static_cast<int>(recvcount), recvtype, comm,
                     CUDA_DEVICE_TAG);

        checked_cuda_memcpy(recv_data, host_recv, recvbytes, cudaMemcpyHostToDevice, stream, comm);

        std::free(host_send);
        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_alltoall(send_data, static_cast<int>(sendcount), sendtype,
                     recv_data, static_cast<int>(recvcount), recvtype, comm,
                     CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_alltoall_ffi,
    mpi_alltoall_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("comm")
);

// --- Bcast FFI (GPU) ---
ffi::Error mpi_bcast_ffi_cuda_impl(
    cudaStream_t stream,
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

    if (COPY_TO_HOST) {
        int dtype_size;
        MPI_Type_size(dtype, &dtype_size);
        size_t count = static_cast<size_t>(dtype_size) * nitems;

        void* host_buf = checked_malloc(count, comm);

        if (rank == static_cast<int>(root)) {
            checked_cuda_memcpy(host_buf, in_data, count, cudaMemcpyDeviceToHost, stream, comm);
        }

        mpi_bcast(host_buf, static_cast<int>(nitems), dtype, static_cast<int>(root), comm,
                  CUDA_DEVICE_TAG);

        if (rank != static_cast<int>(root)) {
            checked_cuda_memcpy(out_data, host_buf, count, cudaMemcpyHostToDevice, stream, comm);
        }

        std::free(host_buf);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        if (rank == static_cast<int>(root)) {
            // Root broadcasts from input buffer
            mpi_bcast(in_data, static_cast<int>(nitems), dtype, static_cast<int>(root), comm,
                      CUDA_DEVICE_TAG);
        } else {
            // Non-root: copy input to output first, then receive in output
            int dtype_size;
            MPI_Type_size(dtype, &dtype_size);
            std::memcpy(out_data, in_data, static_cast<size_t>(nitems) * dtype_size);
            mpi_bcast(out_data, static_cast<int>(nitems), dtype, static_cast<int>(root), comm,
                      CUDA_DEVICE_TAG);
        }
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_bcast_ffi,
    mpi_bcast_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Gather FFI (GPU) ---
ffi::Error mpi_gather_ffi_cuda_impl(
    cudaStream_t stream,
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

    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    if (COPY_TO_HOST) {
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

        checked_cuda_memcpy(host_send, send_data, sendbytes, cudaMemcpyDeviceToHost, stream, comm);

        mpi_gather(host_send, static_cast<int>(sendcount), sendtype,
                   host_recv, static_cast<int>(recvcount), recvtype,
                   static_cast<int>(root), comm, CUDA_DEVICE_TAG);

        if (rank == static_cast<int>(root)) {
            checked_cuda_memcpy(recv_data, host_recv, recvbytes, cudaMemcpyHostToDevice, stream, comm);
        }

        std::free(host_send);
        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_gather(send_data, static_cast<int>(sendcount), sendtype,
                   recv_data, static_cast<int>(recvcount), recvtype,
                   static_cast<int>(root), comm, CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_gather_ffi,
    mpi_gather_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
);

// --- Scatter FFI (GPU) ---
ffi::Error mpi_scatter_ffi_cuda_impl(
    cudaStream_t stream,
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

    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    if (COPY_TO_HOST) {
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

        checked_cuda_memcpy(host_send, send_data, sendbytes, cudaMemcpyDeviceToHost, stream, comm);

        mpi_scatter(host_send, static_cast<int>(sendcount), sendtype,
                    host_recv, static_cast<int>(recvcount), recvtype,
                    static_cast<int>(root), comm, CUDA_DEVICE_TAG);

        checked_cuda_memcpy(recv_data, host_recv, recvbytes, cudaMemcpyHostToDevice, stream, comm);

        std::free(host_send);
        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_scatter(send_data, static_cast<int>(sendcount), sendtype,
                    recv_data, static_cast<int>(recvcount), recvtype,
                    static_cast<int>(root), comm, CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_scatter_ffi,
    mpi_scatter_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("sendcount")
        .Attr<int64_t>("sendtype")
        .Attr<int64_t>("recvcount")
        .Attr<int64_t>("recvtype")
        .Attr<int64_t>("root")
        .Attr<int64_t>("comm")
);

// --- Reduce FFI (GPU) ---
ffi::Error mpi_reduce_ffi_cuda_impl(
    cudaStream_t stream,
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

    if (COPY_TO_HOST) {
        int dtype_size;
        MPI_Type_size(dtype, &dtype_size);
        size_t count = static_cast<size_t>(dtype_size) * nitems;

        void* host_send = checked_malloc(count, comm);
        void* host_recv = checked_malloc(count, comm);

        checked_cuda_memcpy(host_send, send_data, count, cudaMemcpyDeviceToHost, stream, comm);

        mpi_reduce(host_send, host_recv, static_cast<int>(nitems), dtype, op,
                   static_cast<int>(root), comm, CUDA_DEVICE_TAG);

        if (rank == static_cast<int>(root)) {
            checked_cuda_memcpy(recv_data, host_recv, count, cudaMemcpyHostToDevice, stream, comm);
        }

        std::free(host_send);
        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_reduce(send_data, recv_data, static_cast<int>(nitems), dtype, op,
                   static_cast<int>(root), comm, CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_reduce_ffi,
    mpi_reduce_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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

// --- Scan FFI (GPU) ---
ffi::Error mpi_scan_ffi_cuda_impl(
    cudaStream_t stream,
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

    if (COPY_TO_HOST) {
        int dtype_size;
        MPI_Type_size(dtype, &dtype_size);
        size_t count = static_cast<size_t>(dtype_size) * nitems;

        void* host_send = checked_malloc(count, comm);
        void* host_recv = checked_malloc(count, comm);

        checked_cuda_memcpy(host_send, send_data, count, cudaMemcpyDeviceToHost, stream, comm);

        mpi_scan(host_send, host_recv, static_cast<int>(nitems), dtype, op, comm,
                 CUDA_DEVICE_TAG);

        checked_cuda_memcpy(recv_data, host_recv, count, cudaMemcpyHostToDevice, stream, comm);

        std::free(host_send);
        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_scan(send_data, recv_data, static_cast<int>(nitems), dtype, op, comm,
                 CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_scan_ffi,
    mpi_scan_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("op")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Send FFI (GPU) ---
ffi::Error mpi_send_ffi_cuda_impl(
    cudaStream_t stream,
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

    if (COPY_TO_HOST) {
        int dtype_size;
        MPI_Type_size(dtype, &dtype_size);
        size_t count = static_cast<size_t>(dtype_size) * nitems;

        void* host_send = checked_malloc(count, comm);

        checked_cuda_memcpy(host_send, send_data, count, cudaMemcpyDeviceToHost, stream, comm);

        mpi_send(host_send, static_cast<int>(nitems), dtype,
                 static_cast<int>(dest), static_cast<int>(tag), comm,
                 CUDA_DEVICE_TAG);

        std::free(host_send);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_send(send_data, static_cast<int>(nitems), dtype,
                 static_cast<int>(dest), static_cast<int>(tag), comm,
                 CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_send_ffi,
    mpi_send_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("nitems")
        .Attr<int64_t>("dest")
        .Attr<int64_t>("tag")
        .Attr<int64_t>("comm")
        .Attr<int64_t>("dtype")
);

// --- Recv FFI (GPU) ---
ffi::Error mpi_recv_ffi_cuda_impl(
    cudaStream_t stream,
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

    if (COPY_TO_HOST) {
        int dtype_size;
        MPI_Type_size(dtype, &dtype_size);
        size_t count = static_cast<size_t>(dtype_size) * nitems;

        void* host_recv = checked_malloc(count, comm);

        mpi_recv(host_recv, static_cast<int>(nitems), dtype,
                 static_cast<int>(source), static_cast<int>(tag), comm, status,
                 CUDA_DEVICE_TAG);

        checked_cuda_memcpy(recv_data, host_recv, count, cudaMemcpyHostToDevice, stream, comm);

        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_recv(recv_data, static_cast<int>(nitems), dtype,
                 static_cast<int>(source), static_cast<int>(tag), comm, status,
                 CUDA_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_recv_ffi,
    mpi_recv_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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

// --- Sendrecv FFI (GPU) ---
ffi::Error mpi_sendrecv_ffi_cuda_impl(
    cudaStream_t stream,
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
    int64_t status_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Status* status = from_handle<MPI_Status*>(status_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    if (COPY_TO_HOST) {
        int send_dtype_size, recv_dtype_size;
        MPI_Type_size(sendtype, &send_dtype_size);
        MPI_Type_size(recvtype, &recv_dtype_size);

        size_t bytes_send = static_cast<size_t>(send_dtype_size) * sendcount;
        size_t bytes_recv = static_cast<size_t>(recv_dtype_size) * recvcount;

        void* host_send = checked_malloc(bytes_send, comm);
        void* host_recv = checked_malloc(bytes_recv, comm);

        checked_cuda_memcpy(host_send, send_data, bytes_send, cudaMemcpyDeviceToHost, stream, comm);

        mpi_sendrecv(
            host_send, static_cast<int>(sendcount), sendtype,
            static_cast<int>(dest), static_cast<int>(sendtag),
            host_recv, static_cast<int>(recvcount), recvtype,
            static_cast<int>(source), static_cast<int>(recvtag),
            comm, status, CUDA_DEVICE_TAG
        );

        checked_cuda_memcpy(recv_data, host_recv, bytes_recv, cudaMemcpyHostToDevice, stream, comm);

        std::free(host_send);
        std::free(host_recv);
    } else {
        checked_cuda_stream_synchronize(stream, comm);
        mpi_sendrecv(
            send_data, static_cast<int>(sendcount), sendtype,
            static_cast<int>(dest), static_cast<int>(sendtag),
            recv_data, static_cast<int>(recvcount), recvtype,
            static_cast<int>(source), static_cast<int>(recvtag),
            comm, status, CUDA_DEVICE_TAG
        );
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_sendrecv_ffi,
    mpi_sendrecv_ffi_cuda_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::Token>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::Token>()
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
// nanobind module definition
// ============================================================================
NB_MODULE(mpi_xla_bridge_cuda, m) {
    m.doc() = "C++ MPI XLA bridge for CUDA (nanobind) with XLA FFI support";

    // Logging control (uses shared functions from mpi_ops_common.h)
    MPI4JAX_REGISTER_LOGGING(m);

    // Copy-to-host mode control
    m.def("set_copy_to_host", &set_copy_to_host,
          "Enable or disable copy-to-host mode for non-GPU-aware MPI",
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
