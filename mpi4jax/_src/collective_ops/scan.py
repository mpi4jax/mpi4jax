import numpy as _np
from mpi4py import MPI as _MPI

from jax import core
from jax.core import Tracer
from jax.lax import create_token
from jax.interpreters.mlir import custom_call
from jax.core import ShapedArray


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
from ..jax_compat import register_lowering, token_type, Primitive, Token
from ..decorators import (
    translation_rule_cpu,
    translation_rule_cuda,
    translation_rule_xpu,
)
from ..validation import enforce_types
from ..comm import get_default_comm
from ..xla_bridge.device_descriptors import build_scan_descriptor


# The Jax primitive
mpi_scan_p = Primitive("scan_mpi")  # Create the primitive
mpi_scan_impl = default_primitive_impl(mpi_scan_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def scan(x, op, *, comm=None, token=None):
    """Perform a scan operation.

    Arguments:
        x: Array or scalar input to send.
        op (mpi4py.MPI.Op): The reduction operator (e.g :obj:`mpi4py.MPI.SUM`).
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Result of the scan operation.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if prefer_notoken():
        from mpi4jax._src.notoken import scan

        return scan(x, op, comm=comm), token

    if comm is None:
        comm = get_default_comm()

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    return tuple(mpi_scan_p.bind(x, token, op=op, comm=comm))


# This function compiles the operation
@translation_rule_cpu
def mpi_scan_xla_encode_cpu(ctx, x, token, op, comm):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

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
        token_type(),
    ]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(to_mpi_handle(op), _np.uintp),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        token,
    )

    return custom_call(
        b"mpi_scan",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    ).results


def mpi_scan_xla_encode_device(ctx, x, token, op, comm):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

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
        token_type(),
    ]

    operands = (
        x,
        token,
    )

    descriptor = build_scan_descriptor(
        nitems,
        to_mpi_handle(op),
        to_mpi_handle(comm),
        dtype_handle,
    )

    return custom_call(
        b"mpi_scan",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    ).results


mpi_scan_xla_encode_cuda = translation_rule_cuda(mpi_scan_xla_encode_device)
mpi_scan_xla_encode_xpu = translation_rule_xpu(mpi_scan_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_scan_abstract_eval(xs, token, op, comm):
    return (
        ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    ), {effect}


mpi_scan_p.multiple_results = True
mpi_scan_p.def_impl(mpi_scan_impl)
mpi_scan_p.def_effectful_abstract_eval(mpi_scan_abstract_eval)

# assign to the primitive the correct encoder
register_lowering(mpi_scan_p, mpi_scan_xla_encode_cpu, platform="cpu")
register_lowering(mpi_scan_p, mpi_scan_xla_encode_cuda, platform="cuda")
register_lowering(mpi_scan_p, mpi_scan_xla_encode_xpu, platform="xpu")
