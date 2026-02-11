import numpy as _np
from mpi4py import MPI as _MPI

from jax.core import ShapedArray
from jax._src.interpreters.mlir import custom_call
import jaxlib.mlir.ir as ir

from mpi4jax._src.utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
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

from mpi4jax._src.xla_bridge.device_descriptors import build_allgather_descriptor


# Check if FFI-based C++ implementation is available
def _has_ffi_support():
    try:
        from mpi4jax._src.xla_bridge import HAS_CPP_EXT, HAS_FFI_TARGETS

        return HAS_CPP_EXT and HAS_FFI_TARGETS
    except ImportError:
        return False


# The Jax primitive
mpi_allgather_p = Primitive("allgather_mpi")  # Create the primitive
mpi_allgather_impl = default_primitive_impl(mpi_allgather_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def allgather(
    x,
    *,
    comm=None,
    token=NOTSET,
):
    """Perform an allgather operation.

    .. warning::

       ``x`` must have the same shape and dtype on all processes.

    Arguments:
        x: Array or scalar input to send.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Received data.

    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    return mpi_allgather_p.bind(
        x,
        comm=comm,
    )


# FFI-based CPU lowering rule using jax.ffi (new typed API)
def mpi_allgather_xla_encode_cpu_ffi(ctx, sendbuf, comm):
    comm = unpack_hashable(comm)

    sendbuf_aval, *_ = ctx.avals_in
    send_nptype = sendbuf_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dtype = send_type.element_type
    send_dims = send_type.shape

    # compute total number of elements in array
    send_nitems = int(_np.prod(send_dims, dtype=int))
    dtype_handle = int(to_dtype_handle(send_nptype))

    size = comm.Get_size()
    out_shape = (size, *send_dims)

    token = get_token_effect(ctx, ordered_effect)

    out_types = [
        ir.RankedTensorType.get(out_shape, send_dtype),
        token_type(),
    ]

    operands = (sendbuf, token)

    backend_config = {
        "sendcount": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), send_nitems),
        "sendtype": ir.IntegerAttr.get(ir.IntegerType.get_unsigned(64), dtype_handle),
        "recvcount": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), send_nitems),
        "recvtype": ir.IntegerAttr.get(ir.IntegerType.get_unsigned(64), dtype_handle),
        "comm": ir.IntegerAttr.get(
            ir.IntegerType.get_unsigned(64), int(to_mpi_handle(comm))
        ),
    }

    result_obj = custom_call(
        b"mpi_allgather_ffi",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
        api_version=4,
        backend_config=backend_config,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Legacy CPU lowering rule (api_version=0)
def mpi_allgather_xla_encode_cpu_legacy(ctx, sendbuf, comm):
    comm = unpack_hashable(comm)

    sendbuf_aval, *_ = ctx.avals_in
    send_nptype = sendbuf_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dtype = send_type.element_type
    send_dims = send_type.shape

    # compute total number of elements in array
    send_nitems = _np.prod(send_dims, dtype=int)

    size = comm.Get_size()
    out_shape = (size, *send_dims)

    out_types = [
        ir.RankedTensorType.get(out_shape, send_dtype),
        token_type(),
    ]

    token = get_token_effect(ctx, ordered_effect)

    operands = (
        as_mhlo_constant(send_nitems, _np.intc),
        sendbuf,
        as_mhlo_constant(to_dtype_handle(send_nptype), _np.uintp),
        # we only support matching input and output arrays
        as_mhlo_constant(send_nitems, _np.intc),
        as_mhlo_constant(to_dtype_handle(send_nptype), _np.uintp),
        #
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        token,
    )

    result_obj = custom_call(
        b"mpi_allgather",
        result_types=out_types,
        operands=operands,
        # layout matters here, because the first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Choose which CPU lowering to use based on FFI availability
@translation_rule_cpu
def mpi_allgather_xla_encode_cpu(ctx, sendbuf, comm):
    import os

    use_ffi = os.getenv("MPI4JAX_USE_FFI", "true").lower() in ("true", "1", "on")

    if use_ffi and _has_ffi_support():
        return mpi_allgather_xla_encode_cpu_ffi(ctx, sendbuf, comm)
    else:
        return mpi_allgather_xla_encode_cpu_legacy(ctx, sendbuf, comm)


def mpi_allgather_xla_encode_device(ctx, sendbuf, comm):
    comm = unpack_hashable(comm)

    sendbuf_aval, *_ = ctx.avals_in
    send_nptype = sendbuf_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dtype = send_type.element_type
    send_dims = send_type.shape

    # compute total number of elements in send array
    send_nitems = _np.prod(send_dims, dtype=int)
    send_dtype_handle = to_dtype_handle(send_nptype)

    size = comm.Get_size()
    out_shape = (size, *send_dims)

    out_types = [
        ir.RankedTensorType.get(out_shape, send_dtype),
        token_type(),
    ]

    descriptor = build_allgather_descriptor(
        send_nitems,
        send_dtype_handle,
        # we only support matching input and output arrays
        send_nitems,
        send_dtype_handle,
        #
        to_mpi_handle(comm),
    )

    token = get_token_effect(ctx, ordered_effect)

    operands = (sendbuf, token)

    result_obj = custom_call(
        b"mpi_allgather",
        result_types=out_types,
        operands=operands,
        # layout matters here, because the first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        backend_config=descriptor,
        has_side_effect=True,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


mpi_allgather_xla_encode_xpu = translation_rule_xpu(mpi_allgather_xla_encode_device)
mpi_allgather_xla_encode_cuda = translation_rule_cuda(mpi_allgather_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_allgather_abstract_eval(x, comm):
    comm = unpack_hashable(comm)
    size = comm.Get_size()
    out_shape = (size, *x.shape)
    return ShapedArray(out_shape, x.dtype), {ordered_effect}


mpi_allgather_p.def_impl(mpi_allgather_impl)
mpi_allgather_p.def_effectful_abstract_eval(mpi_allgather_abstract_eval)

register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_cpu, platform="cpu")
register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_cuda, platform="cuda")
register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_xpu, platform="xpu")
