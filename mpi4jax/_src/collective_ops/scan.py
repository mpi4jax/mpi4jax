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
        *token_type(),
    ]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(to_mpi_handle(op), _np.uintp),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        token,
    )

    return hlo_custom_call(
        b"mpi_scan",
        out_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_scan_xla_encode_gpu(ctx, x, token, op, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_scan_descriptor

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
        *token_type(),
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

    return hlo_custom_call(
        b"mpi_scan",
        out_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    )


# This function evaluates only the shapes during AST construction
def mpi_scan_abstract_eval(xs, token, op, comm):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    ), {effect}


mpi_scan_p.multiple_results = True
mpi_scan_p.def_impl(mpi_scan_impl)
mpi_scan_p.def_effectful_abstract_eval(mpi_scan_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_scan_p, mpi_scan_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_scan_p, mpi_scan_xla_encode_gpu, platform="cuda")
