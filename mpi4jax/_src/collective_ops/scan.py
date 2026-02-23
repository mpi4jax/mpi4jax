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
mpi_scan_p = Primitive("scan_mpi")  # Create the primitive
mpi_scan_impl = default_primitive_impl(mpi_scan_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def scan(x, op, *, comm=None, token=NOTSET):
    """Perform a scan operation.

    Arguments:
        x: Array or scalar input to send.
        op (mpi4py.MPI.Op): The reduction operator (e.g :obj:`mpi4py.MPI.SUM`).
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Result of the scan operation.
    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    return mpi_scan_p.bind(x, op=op, comm=comm)


def _mpi_scan_xla_encode(ctx, x, op, comm):
    """Common lowering for all platforms using jax.ffi.ffi_lowering."""
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    nitems = _np.prod(x_aval.shape, dtype=_np.int64)
    dtype_handle = _np.int64(to_dtype_handle(x_nptype))

    token = get_token_effect(ctx, ordered_effect)
    operands = (x, token)

    ctx_with_token = ctx.replace(
        avals_in=(*ctx.avals_in, core.abstract_token),
        avals_out=(*ctx.avals_out, core.abstract_token),
    )

    lowering_rule = ffi_lowering(
        "mpi_scan_ffi",
        has_side_effect=True,
    )

    results = lowering_rule(
        ctx_with_token,
        *operands,
        nitems=nitems,
        op=_np.int64(to_mpi_handle(op)),
        comm=_np.int64(to_mpi_handle(comm)),
        dtype=dtype_handle,
    )

    results = list(results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Platform-specific lowering rules (all use the same FFI implementation)
mpi_scan_xla_encode_cpu = translation_rule_cpu(_mpi_scan_xla_encode)
mpi_scan_xla_encode_cuda = translation_rule_cuda(_mpi_scan_xla_encode)
mpi_scan_xla_encode_xpu = translation_rule_xpu(_mpi_scan_xla_encode)


# This function evaluates only the shapes during AST construction
def mpi_scan_abstract_eval(xs, op, comm):
    return ShapedArray(xs.shape, xs.dtype), {ordered_effect}


mpi_scan_p.def_impl(mpi_scan_impl)
mpi_scan_p.def_effectful_abstract_eval(mpi_scan_abstract_eval)

# assign to the primitive the correct encoder
register_lowering(mpi_scan_p, mpi_scan_xla_encode_cpu, platform="cpu")
register_lowering(mpi_scan_p, mpi_scan_xla_encode_cuda, platform="cuda")
register_lowering(mpi_scan_p, mpi_scan_xla_encode_xpu, platform="xpu")
