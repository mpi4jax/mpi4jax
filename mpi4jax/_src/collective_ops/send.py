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
    unpack_hashable,
    wrap_as_hashable,
    as_mhlo_constant,
    get_default_layouts,
    effect,
    prefer_notoken,
)
from ..jax_compat import custom_call, token_type
from ..decorators import (
    translation_rule_cpu,
    translation_rule_gpu,
    translation_rule_xpu,
)
from ..validation import enforce_types
from ..comm import get_default_comm
from ..xla_bridge.device_descriptors import build_send_descriptor


# The Jax primitive
mpi_send_p = Primitive("send_mpi")  # Create the primitive
mpi_send_impl = default_primitive_impl(mpi_send_p)


# This function applies the primitive to an AST
@enforce_types(
    dest=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def send(x, dest, *, tag=0, comm=None, token=None):
    """Perform a send operation.

    Arguments:
        x: Array or scalar input to send.
        dest (int): Rank of the destination MPI process.
        tag (int): Tag of this message.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Token: A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if prefer_notoken():
        from mpi4jax.experimental.notoken import send

        send(x, dest, tag=tag, comm=comm)
        return token

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)
    return mpi_send_p.bind(x, token, dest=dest, tag=tag, comm=comm)


# This function compiles the operation
@translation_rule_cpu
def mpi_send_xla_encode_cpu(ctx, x, token, dest, tag, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = token_type()

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(dest, _np.intc),
        as_mhlo_constant(tag, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        token,
    )

    return custom_call(
        b"mpi_send",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    ).results


def mpi_send_xla_encode_device(ctx, x, token, dest, tag, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = token_type()

    operands = (
        x,
        token,
    )

    descriptor = build_send_descriptor(
        nitems,
        dest,
        tag,
        to_mpi_handle(comm),
        dtype_handle,
    )

    return custom_call(
        b"mpi_send",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    ).results


mpi_send_xla_encode_xpu = translation_rule_xpu(mpi_send_xla_encode_device)
mpi_send_xla_encode_gpu = translation_rule_gpu(mpi_send_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_send_abstract_eval(xs, token, dest, tag, comm):
    return core.abstract_token, {effect}


mpi_send_p.def_impl(mpi_send_impl)
mpi_send_p.def_effectful_abstract_eval(mpi_send_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_send_p, mpi_send_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_send_p, mpi_send_xla_encode_gpu, platform="cuda")
mlir.register_lowering(mpi_send_p, mpi_send_xla_encode_xpu, platform="xpu")
