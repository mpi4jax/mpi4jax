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

from mpi4jax._src.xla_bridge.device_descriptors import build_reduce_descriptor

# The Jax primitive
mpi_reduce_p = Primitive("reduce_mpi")  # Create the primitive
mpi_reduce_impl = default_primitive_impl(mpi_reduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    root=(_np.integer),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def reduce(x, op, root, *, comm=None):
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

    if comm is None:
        comm = get_default_comm()

    rank = comm.Get_rank()

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    res = mpi_reduce_p.bind(x, op=op, root=root, comm=comm)

    if rank != root:
        return x

    return res


# This function compiles the operation
@translation_rule_cpu
def mpi_reduce_xla_encode_cpu(ctx, x, op, root, comm):
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

    # output is only used on root, so prevent memory allocation
    rank = comm.Get_rank()

    if rank != root:
        dims = (0,)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    token = ctx.tokens_in.get(ordered_effect)[0]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(to_mpi_handle(op), _np.uintp),
        as_mhlo_constant(root, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        token,
    )

    result_obj = custom_call(
        b"mpi_reduce",
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


def mpi_reduce_xla_encode_device(ctx, x, op, root, comm):
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

    # output is only used on root, so prevent memory allocation
    rank = comm.Get_rank()

    if rank != root:
        dims = (0,)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    token = ctx.tokens_in.get(ordered_effect)[0]

    operands = (
        x,
        token,
    )

    descriptor = build_reduce_descriptor(
        nitems,
        to_mpi_handle(op),
        root,
        to_mpi_handle(comm),
        dtype_handle,
    )

    result_obj = custom_call(
        b"mpi_reduce",
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


mpi_reduce_xla_encode_xpu = translation_rule_xpu(mpi_reduce_xla_encode_device)
mpi_reduce_xla_encode_gpu = translation_rule_gpu(mpi_reduce_xla_encode_device)


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
mlir.register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_gpu, platform="cuda")
mlir.register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_xpu, platform="xpu")
