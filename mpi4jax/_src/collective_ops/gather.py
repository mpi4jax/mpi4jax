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
mpi_gather_p = Primitive("gather_mpi")  # Create the primitive
mpi_gather_impl = default_primitive_impl(mpi_gather_p)


# This function applies the primitive to an AST
@enforce_types(
    root=(_np.integer),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def gather(
    x,
    root,
    *,
    comm=None,
    token=None,
):
    """Perform a gather operation.

    .. warning::

       ``x`` must have the same shape and dtype on all processes.

    .. warning::

        The shape of the returned data varies between ranks. On the root process,
        it is ``(nproc, *input_shape)``. On all other processes the output is
        identical to the input.

    Arguments:
        x: Array or scalar input to send.
        root (int): Rank of the root MPI process.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data on root process, otherwise unmodified input.
            - A new, modified token, that depends on this operation.
    """
    if token is None:
        token = create_token(x)

    if comm is None:
        comm = get_default_comm()

    rank = comm.Get_rank()
    comm = wrap_as_hashable(comm)

    res, token = mpi_gather_p.bind(
        x,
        token,
        root=root,
        comm=comm,
    )

    if rank != root:
        return (x, token)

    return (res, token)


# This function compiles the operation
@translation_rule_cpu
def mpi_gather_xla_encode_cpu(ctx, x, token, root, comm):
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
    size = comm.Get_size()
    if rank == root:
        out_shape = (size, *dims)
    else:
        out_shape = (0,)

    out_types = [
        ir.RankedTensorType.get(out_shape, dtype),
        *token_type(),
    ]

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

    return hlo_custom_call(
        b"mpi_gather",
        out_types=out_types,
        operands=operands,
        # enforce c order because the first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_gather_xla_encode_gpu(ctx, x, token, root, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_gather_descriptor

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
    size = comm.Get_size()
    if rank == root:
        out_shape = (size, *dims)
    else:
        out_shape = (0,)

    out_types = [
        ir.RankedTensorType.get(out_shape, dtype),
        *token_type(),
    ]

    operands = (
        x,
        token,
    )

    descriptor = build_gather_descriptor(
        nitems,
        dtype_handle,
        # we only support matching input and output arrays
        nitems,
        dtype_handle,
        #
        root,
        to_mpi_handle(comm),
    )

    return hlo_custom_call(
        b"mpi_gather",
        out_types=out_types,
        operands=operands,
        # enforce c order because the first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
        backend_config=descriptor,
    )


# This function evaluates only the shapes during AST construction
def mpi_gather_abstract_eval(x, token, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        out_shape = (size, *x.shape)
    else:
        out_shape = (0,)

    return (
        abstract_arrays.ShapedArray(out_shape, x.dtype),
        core.abstract_token,
    ), {effect}


mpi_gather_p.multiple_results = True
mpi_gather_p.def_impl(mpi_gather_impl)
mpi_gather_p.def_effectful_abstract_eval(mpi_gather_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_gather_p, mpi_gather_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_gather_p, mpi_gather_xla_encode_gpu, platform="cuda")
