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
from mpi4jax._src.decorators import translation_rule_cpu, translation_rule_gpu
from mpi4jax._src.validation import enforce_types
from mpi4jax._src.comm import get_default_comm


# The Jax primitive
mpi_scatter_p = Primitive("scatter_mpi")  # Create the primitive
mpi_scatter_impl = default_primitive_impl(mpi_scatter_p)


# This function applies the primitive to an AST
@enforce_types(
    root=(_np.integer),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def scatter(
    x,
    root,
    *,
    comm=None,
):
    """Perform a scatter operation.

    .. warning::

        Unlike mpi4py's scatter, this returns a *new* array with the received data.

    .. warning::

        The expected shape of the first input varies between ranks. On the root process,
        it is ``(nproc, *input_shape)``. On all other processes, it is ``input_shape``.

    Arguments:
        x: Array or scalar input with the correct shape and dtype. On the root process,
           this contains the data to send, and its first axis must have size ``nproc``.
           On non-root processes, this may contain arbitrary data and will not be
           overwritten.
        root (int): Rank of the root MPI process.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Received data.

    """
    if comm is None:
        comm = get_default_comm()

    rank = comm.Get_rank()
    if rank == root:
        size = comm.Get_size()
        if x.shape[0] != size:
            raise ValueError("Scatter input must have shape (nproc, ...)")

    comm = wrap_as_hashable(comm)

    return mpi_scatter_p.bind(
        x,
        root=root,
        comm=comm,
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_scatter_xla_encode_cpu(ctx, x, root, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    rank = comm.Get_rank()
    if rank == root:
        dims = dims[1:]

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    token = ctx.tokens_in.get(ordered_effect)[0]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(dtype_handle, _np.uintp),
        # we only support matching input and output arrays
        as_mhlo_constant(nitems, _np.intc),
        as_mhlo_constant(dtype_handle, _np.uintp),
        #
        as_mhlo_constant(root, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        token,
    )

    result_obj = custom_call(
        b"mpi_scatter",
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


@translation_rule_gpu
def mpi_scatter_xla_encode_gpu(ctx, x, root, comm):
    from mpi4jax._src.xla_bridge.mpi_xla_bridge_gpu import build_scatter_descriptor

    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    rank = comm.Get_rank()
    if rank == root:
        dims = dims[1:]

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    token = ctx.tokens_in.get(ordered_effect)[0]

    operands = (
        x,
        token,
    )

    descriptor = build_scatter_descriptor(
        nitems,
        dtype_handle,
        # we only support matching input and output arrays
        nitems,
        dtype_handle,
        #
        root,
        to_mpi_handle(comm),
    )

    result_obj = custom_call(
        b"mpi_scatter",
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


# This function evaluates only the shapes during AST construction
def mpi_scatter_abstract_eval(x, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()
    if rank == root:
        out_shape = x.shape[1:]
    else:
        out_shape = x.shape

    return ShapedArray(out_shape, x.dtype), {ordered_effect}


mpi_scatter_p.def_impl(mpi_scatter_impl)
mpi_scatter_p.def_effectful_abstract_eval(mpi_scatter_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_scatter_p, mpi_scatter_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_scatter_p, mpi_scatter_xla_encode_gpu, platform="cuda")
