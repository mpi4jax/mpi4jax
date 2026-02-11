import numpy as _np
from mpi4py import MPI as _MPI

from jax.core import get_aval
from jax.interpreters import ad, batching
from jax.core import ShapedArray


import jaxlib.mlir.ir as ir
from jax._src.interpreters.mlir import custom_call

from mpi4jax._src.utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    to_mpi_ptr,
    unpack_hashable,
    wrap_as_hashable,
    as_mhlo_constant,
    get_default_layouts,
    ordered_effect,
    NOTSET,
    raise_if_token_is_set,
)
from mpi4jax._src.jax_compat import (
    register_lowering,
    token_type,
    get_token_effect,
    set_token_effect,
    Primitive,
)
from mpi4jax._src.decorators import (
    translation_rule_cpu,
    translation_rule_cuda,
    translation_rule_xpu,
)
from mpi4jax._src.validation import enforce_types
from mpi4jax._src.comm import get_default_comm

from mpi4jax._src.xla_bridge.device_descriptors import build_sendrecv_descriptor


# Check if FFI-based C++ implementation is available
def _has_ffi_support():
    try:
        from mpi4jax._src.xla_bridge import HAS_CPP_EXT, HAS_FFI_TARGETS

        return HAS_CPP_EXT and HAS_FFI_TARGETS
    except ImportError:
        return False


# The Jax primitive
mpi_sendrecv_p = Primitive("sendrecv_mpi")  # Create the primitive
mpi_sendrecv_impl = default_primitive_impl(mpi_sendrecv_p)


# This function applies the primitive to an AST
@enforce_types(
    source=_np.integer,
    dest=_np.integer,
    sendtag=_np.integer,
    recvtag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    status=(type(None), _MPI.Status, HashableMPIType),
)
def sendrecv(
    sendbuf,
    recvbuf,
    source,
    dest,
    *,
    sendtag=0,
    recvtag=_MPI.ANY_TAG,
    comm=None,
    status=None,
    token=NOTSET,
):
    """Perform a sendrecv operation.

    .. warning::

        Unlike mpi4py's sendrecv, this returns a *new* array with the received data.

    Arguments:
        sendbuf: Array or scalar input to send.
        recvbuf: Array or scalar input with the correct shape and dtype. This can
           contain arbitrary data and will not be overwritten.
        source (int): Rank of the source MPI process.
        dest (int): Rank of the destination MPI process.
        sendtag (int): Tag of this message for sending.
        recvtag (int): Tag of this message for receiving.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        status (mpi4py.MPI.Status): Status object, can be used for introspection.

    Returns:
        DeviceArray: Received data.

    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    if status is not None:
        status = wrap_as_hashable(status)

    return mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=False,
    )


# FFI-based CPU lowering rule using jax.ffi (new typed API)
def mpi_sendrecv_xla_encode_cpu_ffi(
    ctx,
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    from mpi4jax._src.xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    if _must_transpose:
        raise RuntimeError(
            "sendrecv cannot be used with forward-mode (vjp) autodiff, because "
            "the gradient might be located on a different mpi rank than the "
            "desired one. Use reverse-mode (jvp) differentiation instead."
        )

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    send_aval, recv_aval, *_ = ctx.avals_in
    send_nptype = send_aval.dtype
    recv_nptype = recv_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dims = send_type.shape

    recv_type = ir.RankedTensorType(recvbuf.type)
    recv_dtype = recv_type.element_type
    recv_dims = recv_type.shape

    # compute total number of elements in arrays
    send_nitems = int(_np.prod(send_dims, dtype=int))
    send_dtype_handle = int(to_dtype_handle(send_nptype))

    recv_nitems = int(_np.prod(recv_dims, dtype=int))
    recv_dtype_handle = int(to_dtype_handle(recv_nptype))

    if status is None:
        status_ptr = int(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = int(to_mpi_ptr(status))

    comm_handle = int(to_mpi_handle(comm))

    token = get_token_effect(ctx, ordered_effect)

    # For FFI, we only return the data buffer, not a token
    # The token is handled separately by JAX's effect system
    out_types = [
        ir.RankedTensorType.get(recv_dims, recv_dtype),
    ]

    # Build the operands - just sendbuf for FFI (no token in FFI API)
    operands = (sendbuf,)

    # For FFI API (api_version=4), backend_config must be a dict of ir.Attribute
    # We need to create MLIR attributes for each parameter
    backend_config = {
        "sendcount": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), send_nitems),
        "dest": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(dest)),
        "sendtag": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(sendtag)),
        "sendtype": ir.IntegerAttr.get(
            ir.IntegerType.get_unsigned(64), send_dtype_handle
        ),
        "recvcount": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), recv_nitems),
        "source": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(source)),
        "recvtag": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(recvtag)),
        "recvtype": ir.IntegerAttr.get(
            ir.IntegerType.get_unsigned(64), recv_dtype_handle
        ),
        "comm": ir.IntegerAttr.get(ir.IntegerType.get_unsigned(64), comm_handle),
        "status": ir.IntegerAttr.get(ir.IntegerType.get_unsigned(64), status_ptr),
    }

    # Use custom_call with api_version=4 for FFI
    result_obj = custom_call(
        b"mpi_sendrecv_ffi",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        api_version=4,  # FFI API version
        backend_config=backend_config,
    )

    results = list(result_obj.results)

    # For FFI with effects, we need to still manage the token
    # The token is passed through unchanged since has_side_effect=True
    # ensures proper ordering
    set_token_effect(ctx, ordered_effect, token)

    return results


# Legacy CPU lowering rule (api_version=0)
def mpi_sendrecv_xla_encode_cpu_legacy(
    ctx,
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    from mpi4jax._src.xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    # when performing forward diff, the gradient will follow the sent message.
    # so if you do a sendrecv from rank 0 to 1, the gradient wrt the inputs of rank 0
    # will end up in rank 1.\
    # it's maybe possible to fix this by, at the end of the calculation, bringing back
    # the gradient to the correct rank, but that would require some study.
    if _must_transpose:
        raise RuntimeError(
            "sendrecv cannot be used with forward-mode (vjp) autodiff, because "
            "the gradient might be located on a different mpi rank than the "
            "desired one. Use reverse-mode (jvp) differentiation instead."
        )

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    send_aval, recv_aval, *_ = ctx.avals_in
    send_nptype = send_aval.dtype
    recv_nptype = recv_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dims = send_type.shape

    recv_type = ir.RankedTensorType(recvbuf.type)
    recv_dtype = recv_type.element_type
    recv_dims = recv_type.shape

    # compute total number of elements in arrays
    send_nitems = _np.prod(send_dims, dtype=int)
    send_dtype_handle = to_dtype_handle(send_nptype)

    recv_nitems = _np.prod(recv_dims, dtype=int)
    recv_dtype_handle = to_dtype_handle(recv_nptype)

    out_types = [
        ir.RankedTensorType.get(recv_dims, recv_dtype),
        token_type(),
    ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    token = get_token_effect(ctx, ordered_effect)

    operands = (
        as_mhlo_constant(send_nitems, _np.intc),
        sendbuf,
        as_mhlo_constant(dest, _np.intc),
        as_mhlo_constant(sendtag, _np.intc),
        as_mhlo_constant(send_dtype_handle, _np.uintp),
        as_mhlo_constant(recv_nitems, _np.intc),
        as_mhlo_constant(source, _np.intc),
        as_mhlo_constant(recvtag, _np.intc),
        as_mhlo_constant(recv_dtype_handle, _np.uintp),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(status_ptr, _np.uintp),
        token,
    )

    result_obj = custom_call(
        b"mpi_sendrecv",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Choose which CPU lowering to use based on FFI availability
@translation_rule_cpu
def mpi_sendrecv_xla_encode_cpu(
    ctx,
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    import os

    # Check if we should use FFI (default: True if available)
    use_ffi = os.getenv("MPI4JAX_USE_FFI", "true").lower() in ("true", "1", "on")

    if use_ffi and _has_ffi_support():
        return mpi_sendrecv_xla_encode_cpu_ffi(
            ctx,
            sendbuf,
            recvbuf,
            source,
            dest,
            sendtag,
            recvtag,
            comm,
            status,
            _must_transpose,
        )
    else:
        return mpi_sendrecv_xla_encode_cpu_legacy(
            ctx,
            sendbuf,
            recvbuf,
            source,
            dest,
            sendtag,
            recvtag,
            comm,
            status,
            _must_transpose,
        )


def mpi_sendrecv_xla_encode_device(
    ctx,
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose,
):
    if _must_transpose:
        raise RuntimeError(
            "sendrecv cannot be used with forward-mode (vjp) autodiff, because "
            "the gradient might be located on a different mpi rank than the "
            "desired one. Use reverse-mode (jvp) differentiation instead."
        )

    from mpi4jax._src.xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    send_aval, recv_aval, *_ = ctx.avals_in
    send_nptype = send_aval.dtype
    recv_nptype = recv_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dims = send_type.shape

    recv_type = ir.RankedTensorType(recvbuf.type)
    recv_dtype = recv_type.element_type
    recv_dims = recv_type.shape

    # compute total number of elements in arrays
    send_nitems = _np.prod(send_dims, dtype=int)
    send_dtype_handle = to_dtype_handle(send_nptype)

    recv_nitems = _np.prod(recv_dims, dtype=int)
    recv_dtype_handle = to_dtype_handle(recv_nptype)

    out_types = [
        ir.RankedTensorType.get(recv_dims, recv_dtype),
        token_type(),
    ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    token = get_token_effect(ctx, ordered_effect)

    operands = (
        sendbuf,
        token,
    )

    descriptor = build_sendrecv_descriptor(
        send_nitems,
        dest,
        sendtag,
        send_dtype_handle,
        recv_nitems,
        source,
        recvtag,
        recv_dtype_handle,
        to_mpi_handle(comm),
        status_ptr,
    )

    result_obj = custom_call(
        b"mpi_sendrecv",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


@translation_rule_xpu
def mpi_sendrecv_xla_encode_xpu(
    ctx,
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    return mpi_sendrecv_xla_encode_device(
        ctx,
        sendbuf,
        recvbuf,
        source,
        dest,
        sendtag,
        recvtag,
        comm,
        status,
        _must_transpose,
    )


@translation_rule_cuda
def mpi_sendrecv_xla_encode_cuda(
    ctx,
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    return mpi_sendrecv_xla_encode_device(
        ctx,
        sendbuf,
        recvbuf,
        source,
        dest,
        sendtag,
        recvtag,
        comm,
        status,
        _must_transpose,
    )


# This function evaluates only the shapes during AST construction
def mpi_sendrecv_abstract_eval(
    sendbuf,
    recvbuf,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    return ShapedArray(recvbuf.shape, recvbuf.dtype), {ordered_effect}


def mpi_sendrecv_batch_eval(
    in_args,
    batch_axes,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    sendbuf, recvbuf = in_args

    assert batch_axes[0] == batch_axes[1]
    ax = batch_axes[0]

    res = mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=_must_transpose,
    )
    return res, ax


def mpi_sendrecv_value_and_jvp(
    in_args,
    tan_args,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    sendbuf, recvbuf = in_args
    send_tan, recv_tan = tan_args

    val = mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=_must_transpose,
    )

    jvp = mpi_sendrecv_p.bind(
        send_tan,
        recv_tan,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=not _must_transpose,
    )

    return val, jvp


def mpi_sendrecv_transpose_rule(
    out_tan, *x_args, source, dest, sendtag, recvtag, comm, status, _must_transpose
):
    # swap the sender and receiver
    res = mpi_sendrecv_p.bind(
        out_tan,
        out_tan,
        source=dest,
        dest=source,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=not _must_transpose,
    )
    return (res, ad.Zero(get_aval(res)))


mpi_sendrecv_p.def_impl(mpi_sendrecv_impl)
mpi_sendrecv_p.def_effectful_abstract_eval(mpi_sendrecv_abstract_eval)

batching.primitive_batchers[mpi_sendrecv_p] = mpi_sendrecv_batch_eval

ad.primitive_jvps[mpi_sendrecv_p] = mpi_sendrecv_value_and_jvp
ad.primitive_transposes[mpi_sendrecv_p] = mpi_sendrecv_transpose_rule

# assign to the primitive the correct encoder
register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_cpu, platform="cpu")
register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_cuda, platform="cuda")
register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_xpu, platform="xpu")
