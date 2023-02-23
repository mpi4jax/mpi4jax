import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive, Tracer, Token
from jax.lax import create_token

from jax.interpreters import mlir
import jaxlib.mlir.ir as ir

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    unpack_hashable,
    wrap_as_hashable,
    as_mhlo_constant,
    get_default_layouts,
    effect,
)
from ..jax_compat import hlo_custom_call, token_type
from ..decorators import translation_rule_cpu, translation_rule_gpu
from ..validation import enforce_types
from ..comm import get_default_comm

# The Jax primitive
mpi_allgather_p = Primitive("allgather_mpi")  # Create the primitive
mpi_allgather_impl = default_primitive_impl(mpi_allgather_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def allgather(
    x,
    *,
    comm=None,
    token=None,
):
    """Perform an allgather operation.

    .. warning::

       ``x`` must have the same shape and dtype on all processes.

    Arguments:
        x: Array or scalar input to send.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    return tuple(
        mpi_allgather_p.bind(
            x,
            token,
            comm=comm,
        )
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_allgather_xla_encode_cpu(ctx, sendbuf, token, comm):
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

    return hlo_custom_call(
        b"mpi_allgather",
        out_types=out_types,
        operands=operands,
        # layout matters here, because the first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_allgather_xla_encode_gpu(ctx, sendbuf, token, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_allgather_descriptor

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

    operands = (sendbuf, token)

    return hlo_custom_call(
        b"mpi_allgather",
        out_types=out_types,
        operands=operands,
        # layout matters here, because the first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        backend_config=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_allgather_abstract_eval(x, token, comm):
    comm = unpack_hashable(comm)
    size = comm.Get_size()
    out_shape = (size, *x.shape)
    return (
        abstract_arrays.ShapedArray(out_shape, x.dtype),
        core.abstract_token,
    ), {effect}


mpi_allgather_p.multiple_results = True
mpi_allgather_p.def_impl(mpi_allgather_impl)
mpi_allgather_p.def_effectful_abstract_eval(mpi_allgather_abstract_eval)

mlir.register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_gpu, platform="cuda")
