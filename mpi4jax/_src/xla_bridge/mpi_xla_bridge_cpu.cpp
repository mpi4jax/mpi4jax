// nanobind bindings for mpi4jax CPU custom call targets using the XLA FFI API
// This module implements all MPI operations for CPU using the shared MPI wrappers
// from mpi_ops_common.h

#include "mpi_ops_common.h"

namespace ffi = xla::ffi;

namespace mpi4jax {

// Device tag for CPU (empty string = no tag in debug output)
constexpr char CPU_DEVICE_TAG[] = "";

// ============================================================================
// FFI Handler implementations (attribute-based API)
// ============================================================================

// Barrier FFI implementation
ffi::Error mpi_barrier_ffi_impl(
    ffi::Token token_in,
    ffi::Result<ffi::Token> token_out,
    int64_t comm_handle
) {
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    mpi_barrier(comm, CPU_DEVICE_TAG);
    return ffi::Error::Success();
}

// Allgather FFI implementation
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
                  recv_data, static_cast<int>(recvcount), recvtype, comm,
                  CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Allreduce FFI implementation
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

    mpi_allreduce(send_data, recv_data, static_cast<int>(nitems), dtype, op, comm,
                  CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Alltoall FFI implementation
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
                 recv_data, static_cast<int>(recvcount), recvtype, comm,
                 CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Bcast FFI implementation
ffi::Error mpi_bcast_ffi_impl(
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

    void* in_data = buf.untyped_data();
    void* out_data = out->untyped_data();

    // Get current rank to handle root specially
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == static_cast<int>(root)) {
        // Root rank: broadcast directly from input buffer
        // (output buffer has size 0 on root to save memory)
        mpi_bcast(in_data, static_cast<int>(nitems), dtype, static_cast<int>(root), comm,
                  CPU_DEVICE_TAG);
    } else {
        // Non-root ranks: copy input to output first, then broadcast in-place
        int dtype_size;
        MPI_Type_size(dtype, &dtype_size);
        std::memcpy(out_data, in_data, static_cast<size_t>(nitems) * dtype_size);

        mpi_bcast(out_data, static_cast<int>(nitems), dtype, static_cast<int>(root), comm,
                  CPU_DEVICE_TAG);
    }

    return ffi::Error::Success();
}

// Gather FFI implementation
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
               static_cast<int>(root), comm, CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Scatter FFI implementation
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
                static_cast<int>(root), comm, CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Reduce FFI implementation
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
               static_cast<int>(root), comm, CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Scan FFI implementation
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

    mpi_scan(send_data, recv_data, static_cast<int>(nitems), dtype, op, comm,
             CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Send FFI implementation
ffi::Error mpi_send_ffi_impl(
    ffi::AnyBuffer buf,
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

    void* data = buf.untyped_data();

    mpi_send(data, static_cast<int>(nitems), dtype,
             static_cast<int>(dest), static_cast<int>(tag), comm,
             CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Recv FFI implementation
ffi::Error mpi_recv_ffi_impl(
    ffi::Token token_in,
    ffi::Result<ffi::AnyBuffer> buf,
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
    MPI_Status* status = reinterpret_cast<MPI_Status*>(status_handle);

    void* data = buf->untyped_data();

    mpi_recv(data, static_cast<int>(nitems), dtype,
             static_cast<int>(source), static_cast<int>(tag), comm, status,
             CPU_DEVICE_TAG);

    return ffi::Error::Success();
}

// Sendrecv FFI implementation
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
    int64_t status_handle
) {
    MPI_Datatype sendtype = from_handle<MPI_Datatype>(sendtype_handle);
    MPI_Datatype recvtype = from_handle<MPI_Datatype>(recvtype_handle);
    MPI_Comm comm = from_handle<MPI_Comm>(comm_handle);
    MPI_Status* status = reinterpret_cast<MPI_Status*>(status_handle);

    void* send_data = sendbuf.untyped_data();
    void* recv_data = recvbuf->untyped_data();

    mpi_sendrecv(
        send_data, static_cast<int>(sendcount), sendtype,
        static_cast<int>(dest), static_cast<int>(sendtag),
        recv_data, static_cast<int>(recvcount), recvtype,
        static_cast<int>(source), static_cast<int>(recvtag),
        comm, status, CPU_DEVICE_TAG
    );

    return ffi::Error::Success();
}

// ============================================================================
// FFI Handler symbol definitions (using XLA_FFI_DEFINE_HANDLER_SYMBOL)
// ============================================================================

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_barrier_ffi,
    mpi_barrier_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Token>()
        .Ret<ffi::Token>()
        .Attr<int64_t>("comm")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_allgather_ffi,
    mpi_allgather_ffi_impl,
    ffi::Ffi::Bind()
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_allreduce_ffi,
    mpi_allreduce_ffi_impl,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_alltoall_ffi,
    mpi_alltoall_ffi_impl,
    ffi::Ffi::Bind()
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_bcast_ffi,
    mpi_bcast_ffi_impl,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_gather_ffi,
    mpi_gather_ffi_impl,
    ffi::Ffi::Bind()
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_scatter_ffi,
    mpi_scatter_ffi_impl,
    ffi::Ffi::Bind()
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_reduce_ffi,
    mpi_reduce_ffi_impl,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_scan_ffi,
    mpi_scan_ffi_impl,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_send_ffi,
    mpi_send_ffi_impl,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_recv_ffi,
    mpi_recv_ffi_impl,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mpi_sendrecv_ffi,
    mpi_sendrecv_ffi_impl,
    ffi::Ffi::Bind()
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
// Nanobind module definition
// ============================================================================
NB_MODULE(mpi_xla_bridge_cpu, m) {
    m.doc() = "mpi4jax CPU XLA bridge using FFI API";

    // Register logging functions
    MPI4JAX_REGISTER_LOGGING(m);

    // Register MPI_STATUS_IGNORE address
    m.attr("MPI_STATUS_IGNORE_ADDR") = get_mpi_status_ignore_addr();

    // Register MPI ABI information for compatibility checking
    auto abi_info = get_mpi_abi_info();
    nb::dict mpi_abi;
    mpi_abi["sizeof_comm"] = abi_info.sizeof_comm;
    mpi_abi["sizeof_datatype"] = abi_info.sizeof_datatype;
    mpi_abi["sizeof_op"] = abi_info.sizeof_op;
    mpi_abi["sizeof_status"] = abi_info.sizeof_status;
    mpi_abi["comm_world_handle"] = abi_info.comm_world_handle;
    mpi_abi["mpi_library_version"] = abi_info.mpi_library_version;
    m.attr("MPI_ABI_INFO") = mpi_abi;

    // Register FFI targets
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
