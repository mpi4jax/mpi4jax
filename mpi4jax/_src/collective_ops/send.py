import numpy as _np
from mpi4py import MPI as _MPI

from jax import core
from jax.ffi import ffi_lowering

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
mpi_send_p = Primitive("send_mpi")  # Create the primitive
mpi_send_impl = default_primitive_impl(mpi_send_p)


# This function applies the primitive to an AST
@enforce_types(
    dest=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def send(
    x,
    dest,
    *,
    tag=0,
    comm=None,
    token=NOTSET,
):
    """Perform a send operation.

    Arguments:
        x: Array or scalar input to send.
        dest (int): Rank of the destination MPI process.
        tag (int): Tag of this message.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)
    return mpi_send_p.bind(x, dest=dest, tag=tag, comm=comm)


def _mpi_send_xla_encode(ctx, x, dest, tag, comm):
    """Common lowering for all platforms using FFI with attributes."""
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    nitems = _np.prod(x_aval.shape, dtype=_np.int64)
    dtype_handle = _np.int64(to_dtype_handle(x_nptype))

    token = get_token_effect(ctx, ordered_effect)
    operands = (x, token)

    ctx_with_token = ctx.replace(
        avals_in=(*ctx.avals_in, core.abstract_token),
        avals_out=(core.abstract_token,),
    )

    lowering_rule = ffi_lowering(
        "mpi_send_ffi",
        has_side_effect=True,
    )

    results = lowering_rule(
        ctx_with_token,
        *operands,
        nitems=nitems,
        dest=_np.int64(dest),
        tag=_np.int64(tag),
        comm=_np.int64(to_mpi_handle(comm)),
        dtype=dtype_handle,
    )

    results = list(results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Platform-specific lowering rules (all use the same FFI implementation)
mpi_send_xla_encode_cpu = translation_rule_cpu(_mpi_send_xla_encode)
mpi_send_xla_encode_cuda = translation_rule_cuda(_mpi_send_xla_encode)
mpi_send_xla_encode_xpu = translation_rule_xpu(_mpi_send_xla_encode)


# This function evaluates only the shapes during AST construction
def mpi_send_abstract_eval(xs, dest, tag, comm):
    return (), {ordered_effect}


mpi_send_p.multiple_results = True
mpi_send_p.def_impl(mpi_send_impl)
mpi_send_p.def_effectful_abstract_eval(mpi_send_abstract_eval)

# assign to the primitive the correct encoder
register_lowering(mpi_send_p, mpi_send_xla_encode_cpu, platform="cpu")
register_lowering(mpi_send_p, mpi_send_xla_encode_cuda, platform="cuda")
register_lowering(mpi_send_p, mpi_send_xla_encode_xpu, platform="xpu")
