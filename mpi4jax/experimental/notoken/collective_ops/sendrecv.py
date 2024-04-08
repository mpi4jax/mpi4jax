import numpy as _np
from mpi4py import MPI as _MPI

from jax.lax import create_token
from jax.core import Primitive
from jax.interpreters import ad, batching

from jax.interpreters import mlir
import jaxlib.mlir.ir as ir

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
)
from mpi4jax._src.jax_compat import custom_call, token_type, ShapedArray
from mpi4jax._src.decorators import (
    translation_rule_cpu,
    translation_rule_gpu,
    translation_rule_xpu,
)
from mpi4jax._src.validation import enforce_types
from mpi4jax._src.comm import get_default_comm

from mpi4jax._src.xla_bridge.device_descriptors import build_sendrecv_descriptor


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
    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    if status is not None:
        status = wrap_as_hashable(status)

    token = create_token()

    return mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        token,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=False,
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_sendrecv_xla_encode_cpu(
    ctx,
    sendbuf,
    recvbuf,
    token,
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

    del token  # unused

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
        *token_type(),
    ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    token = ctx.tokens_in.get(ordered_effect)[0]

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
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))

    return results


def mpi_sendrecv_xla_encode_device(
    ctx,
    sendbuf,
    recvbuf,
    token,
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

    del token  # unused

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
        *token_type(),
    ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    token = ctx.tokens_in.get(ordered_effect)[0]

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
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))

    return results


@translation_rule_xpu
def mpi_sendrecv_xla_encode_xpu(
    ctx,
    sendbuf,
    recvbuf,
    token,
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
        token,
        source,
        dest,
        sendtag,
        recvtag,
        comm,
        status,
        _must_transpose,
    )


@translation_rule_gpu
def mpi_sendrecv_xla_encode_gpu(
    ctx,
    sendbuf,
    recvbuf,
    token,
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
        token,
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
    token,
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
    sendbuf, recvbuf, token = in_args

    assert batch_axes[0] == batch_axes[1]
    ax = batch_axes[0]

    res = mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        token,
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
    sendbuf, recvbuf, token = in_args
    send_tan, recv_tan, _ = tan_args

    val = mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        token,
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
        token,
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
    _, _, token = x_args

    # swap the sender and receiver
    res = mpi_sendrecv_p.bind(
        out_tan,
        out_tan,
        token,
        source=dest,
        dest=source,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=not _must_transpose,
    )
    return (res, ad.Zero.from_value(res), ad.Zero.from_value(token))


mpi_sendrecv_p.def_impl(mpi_sendrecv_impl)
mpi_sendrecv_p.def_effectful_abstract_eval(mpi_sendrecv_abstract_eval)

batching.primitive_batchers[mpi_sendrecv_p] = mpi_sendrecv_batch_eval

ad.primitive_jvps[mpi_sendrecv_p] = mpi_sendrecv_value_and_jvp
ad.primitive_transposes[mpi_sendrecv_p] = mpi_sendrecv_transpose_rule

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_gpu, platform="cuda")
mlir.register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_xpu, platform="xpu")
