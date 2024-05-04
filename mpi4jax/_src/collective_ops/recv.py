import numpy as _np
from mpi4py import MPI as _MPI

from jax import core
from jax.core import Primitive, Tracer, Token
from jax.lax import create_token

from jax.interpreters import mlir
import jaxlib.mlir.ir as ir

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    to_mpi_ptr,
    unpack_hashable,
    wrap_as_hashable,
    as_mhlo_constant,
    get_default_layouts,
    effect,
    prefer_notoken,
)
from ..jax_compat import custom_call, token_type, ShapedArray
from ..decorators import (
    translation_rule_cpu,
    translation_rule_gpu,
    translation_rule_xpu,
)
from ..validation import enforce_types
from ..comm import get_default_comm

from ..xla_bridge.device_descriptors import build_recv_descriptor

# The Jax primitive
mpi_recv_p = Primitive("recv_mpi")  # Create the primitive
mpi_recv_impl = default_primitive_impl(mpi_recv_p)


# This function applies the primitive to an AST
@enforce_types(
    source=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    status=(type(None), _MPI.Status, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def recv(
    x,
    source=_MPI.ANY_SOURCE,
    *,
    tag=_MPI.ANY_TAG,
    comm=None,
    status=None,
    token=None,
):
    """Perform a recv (receive) operation.

    .. warning::

        Unlike mpi4py's recv, this returns a *new* array with the received data.

    Arguments:
        x: Array or scalar input with the correct shape and dtype. This can contain
           arbitrary data and will not be overwritten.
        source (int): Rank of the source MPI process.
        tag (int): Tag of this message.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        status (mpi4py.MPI.Status): Status object, can be used for introspection.
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if prefer_notoken():
        from mpi4jax.experimental.notoken import recv

        return recv(x, source, tag=tag, status=status, comm=comm), token

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    if status is not None:
        status = wrap_as_hashable(status)

    return tuple(
        mpi_recv_p.bind(x, token, source=source, tag=tag, comm=comm, status=status)
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_recv_xla_encode_cpu(ctx, x, token, source, tag, comm, status):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        as_mhlo_constant(source, _np.intc),
        as_mhlo_constant(tag, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        as_mhlo_constant(status_ptr, _np.uintp),
        token,
    )

    return custom_call(
        b"mpi_recv",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    ).results


def mpi_recv_xla_encode_device(ctx, x, token, source, tag, comm, status):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    operands = (token,)

    descriptor = build_recv_descriptor(
        nitems,
        source,
        tag,
        to_mpi_handle(comm),
        dtype_handle,
        status_ptr,
    )

    return custom_call(
        b"mpi_recv",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    ).results


mpi_recv_xla_encode_xpu = translation_rule_xpu(mpi_recv_xla_encode_device)
mpi_recv_xla_encode_gpu = translation_rule_gpu(mpi_recv_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_recv_abstract_eval(xs, token, source, tag, comm, status):
    return (
        ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    ), {effect}


mpi_recv_p.multiple_results = True
mpi_recv_p.def_impl(mpi_recv_impl)
mpi_recv_p.def_effectful_abstract_eval(mpi_recv_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_recv_p, mpi_recv_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_recv_p, mpi_recv_xla_encode_gpu, platform="cuda")
mlir.register_lowering(mpi_recv_p, mpi_recv_xla_encode_xpu, platform="xpu")
