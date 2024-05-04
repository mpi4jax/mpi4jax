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

from mpi4jax._src.xla_bridge.device_descriptors import build_alltoall_descriptor


# The Jax primitive
mpi_alltoall_p = Primitive("alltoall_mpi")  # Create the primitive
mpi_alltoall_impl = default_primitive_impl(mpi_alltoall_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def alltoall(
    x,
    *,
    comm=None,
):
    """Perform an alltoall operation.

    Arguments:
        x: Array input to send. First axis must have size ``nproc``.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Received data.

    """
    if comm is None:
        comm = get_default_comm()

    size = comm.Get_size()
    if x.shape[0] != size:
        raise ValueError("Alltoall input must have shape (nproc, ...)")

    comm = wrap_as_hashable(comm)

    return mpi_alltoall_p.bind(
        x,
        comm=comm,
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_alltoall_xla_encode_cpu(ctx, x, comm):
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

    token = ctx.tokens_in.get(ordered_effect)[0]

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

    result_obj = custom_call(
        b"mpi_alltoall",
        result_types=out_types,
        operands=operands,
        # force c order because first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))

    return results


def mpi_alltoall_xla_encode_device(ctx, x, comm):
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

    token = ctx.tokens_in.get(ordered_effect)[0]

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

    result_obj = custom_call(
        b"mpi_alltoall",
        result_types=out_types,
        operands=operands,
        # force c order because first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
        backend_config=descriptor,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    ctx.set_tokens_out(mlir.TokenSet({ordered_effect: (token,)}))
    return results


mpi_alltoall_xla_encode_xpu = translation_rule_xpu(mpi_alltoall_xla_encode_device)
mpi_alltoall_xla_encode_gpu = translation_rule_gpu(mpi_alltoall_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_alltoall_abstract_eval(xs, comm):
    return ShapedArray(xs.shape, xs.dtype), {ordered_effect}


mpi_alltoall_p.def_impl(mpi_alltoall_impl)
mpi_alltoall_p.def_effectful_abstract_eval(mpi_alltoall_abstract_eval)

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_gpu, platform="cuda")
mlir.register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_xpu, platform="xpu")
