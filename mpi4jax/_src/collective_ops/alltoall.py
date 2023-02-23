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
mpi_alltoall_p = Primitive("alltoall_mpi")  # Create the primitive
mpi_alltoall_impl = default_primitive_impl(mpi_alltoall_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def alltoall(
    x,
    *,
    comm=None,
    token=None,
):
    """Perform an alltoall operation.

    Arguments:
        x: Array input to send. First axis must have size ``nproc``.
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

    size = comm.Get_size()
    if x.shape[0] != size:
        raise ValueError("Alltoall input must have shape (nproc, ...)")

    comm = wrap_as_hashable(comm)

    return tuple(
        mpi_alltoall_p.bind(
            x,
            token,
            comm=comm,
        )
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_alltoall_xla_encode_cpu(ctx, x, token, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    size = comm.Get_size()
    assert dims[0] == size
    nitems_per_proc = _np.prod(dims[1:], dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    operands = (
        as_mhlo_constant(nitems_per_proc, _np.intc),
        x,
        as_mhlo_constant(dtype_handle, _np.uintp),
        # we only support matching input and output arrays
        as_mhlo_constant(nitems_per_proc, _np.intc),
        as_mhlo_constant(dtype_handle, _np.uintp),
        #
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        token,
    )

    return hlo_custom_call(
        b"mpi_alltoall",
        out_types=out_types,
        operands=operands,
        # force c order because first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_alltoall_xla_encode_gpu(ctx, x, token, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_alltoall_descriptor

    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    size = comm.Get_size()
    assert dims[0] == size
    nitems_per_proc = _np.prod(dims[1:], dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    operands = (
        x,
        token,
    )

    descriptor = build_alltoall_descriptor(
        nitems_per_proc,
        dtype_handle,
        # we only support matching input and output arrays
        nitems_per_proc,
        dtype_handle,
        #
        to_mpi_handle(comm),
    )

    return hlo_custom_call(
        b"mpi_alltoall",
        out_types=out_types,
        operands=operands,
        # force c order because first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
        backend_config=descriptor,
    )


# This function evaluates only the shapes during AST construction
def mpi_alltoall_abstract_eval(xs, token, comm):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    ), {effect}


mpi_alltoall_p.multiple_results = True
mpi_alltoall_p.def_impl(mpi_alltoall_impl)
mpi_alltoall_p.def_effectful_abstract_eval(mpi_alltoall_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_gpu, platform="cuda")
