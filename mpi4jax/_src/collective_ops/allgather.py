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
from mpi4jax._src.utils import get_default_comm


# The Jax primitive
mpi_allgather_p = Primitive("allgather_mpi")  # Create the primitive
mpi_allgather_impl = default_primitive_impl(mpi_allgather_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def allgather(
    x,
    *,
    comm=None,
    token=NOTSET,
):
    """Perform an allgather operation.

    .. warning::

       ``x`` must have the same shape and dtype on all processes.

    Arguments:
        x: Array or scalar input to send.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Received data.

    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    return mpi_allgather_p.bind(
        x,
        comm=comm,
    )


def _mpi_allgather_xla_encode(ctx, sendbuf, comm):
    """Common lowering for all platforms using jax.ffi.ffi_lowering."""
    comm = unpack_hashable(comm)

    sendbuf_aval, *_ = ctx.avals_in
    send_nptype = sendbuf_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dtype = send_type.element_type
    send_dims = send_type.shape

    # compute total number of elements in array
    send_nitems = _np.prod(send_dims, dtype=_np.int64)
    dtype_handle = _np.int64(to_dtype_handle(send_nptype))

    size = comm.Get_size()
    out_shape = (size, *send_dims)

    token = get_token_effect(ctx, ordered_effect)

    out_types = [
        ir.RankedTensorType.get(out_shape, send_dtype),
        token_type(),
    ]

    operands = (sendbuf, token)

    # layout matters here, because the first axis is special
    lowering_rule = ffi_lowering(
        "mpi_allgather_ffi",
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        result_types=out_types,
        has_side_effect=True,
        skip_ffi_layout_processing=True,
    )

    results = lowering_rule(
        ctx,
        *operands,
        sendcount=send_nitems,
        sendtype=dtype_handle,
        recvcount=send_nitems,
        recvtype=dtype_handle,
        comm=_np.int64(to_mpi_handle(comm)),
    )

    results = list(results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Platform-specific lowering rules (all use the same FFI implementation)
mpi_allgather_xla_encode_cpu = translation_rule_cpu(_mpi_allgather_xla_encode)
mpi_allgather_xla_encode_cuda = translation_rule_cuda(_mpi_allgather_xla_encode)
mpi_allgather_xla_encode_xpu = translation_rule_xpu(_mpi_allgather_xla_encode)


# This function evaluates only the shapes during AST construction
def mpi_allgather_abstract_eval(x, comm):
    comm = unpack_hashable(comm)
    size = comm.Get_size()
    out_shape = (size, *x.shape)
    return ShapedArray(out_shape, x.dtype), {ordered_effect}


mpi_allgather_p.def_impl(mpi_allgather_impl)
mpi_allgather_p.def_effectful_abstract_eval(mpi_allgather_abstract_eval)

register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_cpu, platform="cpu")
register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_cuda, platform="cuda")
register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_xpu, platform="xpu")
