import numpy as _np
from mpi4py import MPI as _MPI

from jax.ffi import ffi_lowering
from jax.core import ShapedArray
import jaxlib.mlir.ir as ir

from mpi4jax._src.utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    unpack_hashable,
    wrap_as_hashable,
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


# The Jax primitive
mpi_reduce_p = Primitive("reduce_mpi")  # Create the primitive
mpi_reduce_impl = default_primitive_impl(mpi_reduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    root=(_np.integer),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def reduce(x, op, root, *, comm=None, token=NOTSET):
    """Perform a reduce operation.

    Arguments:
        x: Array or scalar input to send.
        op (mpi4py.MPI.Op): The reduction operator (e.g :obj:`mpi4py.MPI.SUM`).
        root (int): Rank of the root MPI process.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Result of the reduce operation on root process, otherwise
            unmodified input.
    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    rank = comm.Get_rank()

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    res = mpi_reduce_p.bind(x, op=op, root=root, comm=comm)

    if rank != root:
        return x

    return res


def _mpi_reduce_xla_encode(ctx, x, op, root, comm):
    """Common lowering for all platforms using jax.ffi.ffi_lowering."""
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=_np.int64)
    dtype_handle = _np.int64(to_dtype_handle(x_nptype))

    # output is only used on root, so prevent memory allocation
    rank = comm.Get_rank()

    if rank != root:
        out_dims = (0,)
    else:
        out_dims = dims

    token = get_token_effect(ctx, ordered_effect)

    out_types = [
        ir.RankedTensorType.get(out_dims, dtype),
        token_type(),
    ]

    operands = (x, token)

    lowering_rule = ffi_lowering(
        "mpi_reduce_ffi",
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        result_types=out_types,
        has_side_effect=True,
        skip_ffi_layout_processing=True,
    )

    results = lowering_rule(
        ctx,
        *operands,
        nitems=nitems,
        op=_np.int64(to_mpi_handle(op)),
        root=_np.int64(root),
        comm=_np.int64(to_mpi_handle(comm)),
        dtype=dtype_handle,
    )

    results = list(results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Platform-specific lowering rules (all use the same FFI implementation)
mpi_reduce_xla_encode_cpu = translation_rule_cpu(_mpi_reduce_xla_encode)
mpi_reduce_xla_encode_cuda = translation_rule_cuda(_mpi_reduce_xla_encode)
mpi_reduce_xla_encode_xpu = translation_rule_xpu(_mpi_reduce_xla_encode)


# This function evaluates only the shapes during AST construction
def mpi_reduce_abstract_eval(xs, op, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()

    if rank != root:
        dims = (0,)
    else:
        dims = xs.shape

    return ShapedArray(dims, xs.dtype), {ordered_effect}


mpi_reduce_p.def_impl(mpi_reduce_impl)
mpi_reduce_p.def_effectful_abstract_eval(mpi_reduce_abstract_eval)

# assign to the primitive the correct encoder
register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_cpu, platform="cpu")
register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_cuda, platform="cuda")
register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_xpu, platform="xpu")
