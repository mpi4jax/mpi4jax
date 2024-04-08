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

from mpi4jax._src.xla_bridge.device_descriptors import build_bcast_descriptor


# The Jax primitive
mpi_bcast_p = Primitive("bcast_mpi")  # Create the primitive
mpi_bcast_impl = default_primitive_impl(mpi_bcast_p)


# This function applies the primitive to an AST
@enforce_types(
    root=(_np.integer),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def bcast(x, root, *, comm=None):
    """Perform a bcast (broadcast) operation.

    .. warning::

        Unlike mpi4py's bcast, this returns a *new* array with the received data.

    Arguments:
        x: Array or scalar input. Data is only read on root process. On non-root
           processes, this is used to determine the shape and dtype of the result.
        root (int): The process to use as source.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Received data.

    """
    if comm is None:
        comm = get_default_comm()

    rank = comm.Get_rank()

    comm = wrap_as_hashable(comm)
    res = mpi_bcast_p.bind(x, root=root, comm=comm)

    if rank == root:
        return x

    return res


# This function compiles the operation
@translation_rule_cpu
def mpi_bcast_xla_encode_cpu(ctx, x, root, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    # output is not used on root, so prevent memory allocation
    rank = comm.Get_rank()
    if rank == root:
        dims = (0,)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    token = ctx.tokens_in.get(ordered_effect)[0]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(root, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        token,
    )

    result_obj = custom_call(
        b"mpi_bcast",
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


def mpi_bcast_xla_encode_device(ctx, x, root, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    # output is not used on root, so prevent memory allocation
    rank = comm.Get_rank()
    if rank == root:
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

    descriptor = build_bcast_descriptor(
        nitems,
        root,
        to_mpi_handle(comm),
        dtype_handle,
    )

    result_obj = custom_call(
        b"mpi_bcast",
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


mpi_bcast_xla_encode_xpu = translation_rule_xpu(mpi_bcast_xla_encode_device)
mpi_bcast_xla_encode_gpu = translation_rule_gpu(mpi_bcast_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_bcast_abstract_eval(xs, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()

    if rank == root:
        dims = (0,)
    else:
        dims = xs.shape

    return ShapedArray(dims, xs.dtype), {ordered_effect}


mpi_bcast_p.def_impl(mpi_bcast_impl)
mpi_bcast_p.def_effectful_abstract_eval(mpi_bcast_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_bcast_p, mpi_bcast_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_bcast_p, mpi_bcast_xla_encode_gpu, platform="cuda")
mlir.register_lowering(mpi_bcast_p, mpi_bcast_xla_encode_xpu, platform="xpu")
