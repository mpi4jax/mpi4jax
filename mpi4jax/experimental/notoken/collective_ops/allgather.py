import numpy as _np
from mpi4py import MPI as _MPI

from jax.core import Primitive

from jax.interpreters import mlir
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
)
from mpi4jax._src.jax_compat import custom_call, token_type, ShapedArray
from mpi4jax._src.decorators import (
    translation_rule_cpu,
    translation_rule_gpu,
    translation_rule_xpu,
)
from mpi4jax._src.validation import enforce_types
from mpi4jax._src.comm import get_default_comm

from mpi4jax._src.xla_bridge.device_descriptors import build_allgather_descriptor

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
    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    return mpi_allgather_p.bind(
        x,
        comm=comm,
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_allgather_xla_encode_cpu(ctx, sendbuf, comm):
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
        *token_type(),
    ]

    token = ctx.tokens_in.get(ordered_effect)[0]

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
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))

    return results


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
        *token_type(),
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

    token = ctx.tokens_in.get(ordered_effect)[0]

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
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))

    return results


mpi_allgather_xla_encode_xpu = translation_rule_xpu(mpi_allgather_xla_encode_device)
mpi_allgather_xla_encode_gpu = translation_rule_gpu(mpi_allgather_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_allgather_abstract_eval(x, comm):
    comm = unpack_hashable(comm)
    size = comm.Get_size()
    out_shape = (size, *x.shape)
    return ShapedArray(out_shape, x.dtype), {ordered_effect}


mpi_allgather_p.def_impl(mpi_allgather_impl)
mpi_allgather_p.def_effectful_abstract_eval(mpi_allgather_abstract_eval)

mlir.register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_gpu, platform="cuda")
mlir.register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_xpu, platform="xpu")
