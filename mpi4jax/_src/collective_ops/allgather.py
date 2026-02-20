import numpy as _np
from mpi4py import MPI as _MPI

from jax import core
from jax.ffi import ffi_lowering
from jax.core import ShapedArray

from mpi4jax._src.utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    unpack_hashable,
    wrap_as_hashable,
    ordered_effect,
    NOTSET,
    raise_if_token_is_set,
)
from mpi4jax._src.jax_compat import (
    register_lowering,
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

    send_nitems = _np.prod(sendbuf_aval.shape, dtype=_np.int64)
    dtype_handle = _np.int64(to_dtype_handle(send_nptype))

    token = get_token_effect(ctx, ordered_effect)
    operands = (sendbuf, token)

    ctx_with_token = ctx.replace(
        avals_in=(*ctx.avals_in, core.abstract_token),
        avals_out=(*ctx.avals_out, core.abstract_token),
    )

    lowering_rule = ffi_lowering(
        "mpi_allgather_ffi",
        has_side_effect=True,
    )

    results = lowering_rule(
        ctx_with_token,
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
