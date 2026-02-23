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
    token=NOTSET,
):
    """Perform an alltoall operation.

    Arguments:
        x: Array input to send. First axis must have size ``nproc``.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Received data.

    """
    raise_if_token_is_set(token)

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


def _mpi_alltoall_xla_encode(ctx, x, comm):
    """Common lowering for all platforms using jax.ffi.ffi_lowering."""
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    # compute total number of elements in array
    size = comm.Get_size()
    assert x_aval.shape[0] == size
    nitems_per_proc = _np.prod(x_aval.shape[1:], dtype=_np.int64)
    dtype_handle = _np.int64(to_dtype_handle(x_nptype))

    token = get_token_effect(ctx, ordered_effect)
    operands = (x, token)

    # Extend ctx to include token in avals (needed for ffi_lowering to derive
    # correct layouts and result types from abstract values)
    ctx_with_token = ctx.replace(
        avals_in=(*ctx.avals_in, core.abstract_token),
        avals_out=(*ctx.avals_out, core.abstract_token),
    )

    # ffi_lowering will derive operand_layouts, result_layouts, and result_types
    # from ctx_with_token.avals_in/out automatically
    lowering_rule = ffi_lowering(
        "mpi_alltoall_ffi",
        has_side_effect=True,
    )

    results = lowering_rule(
        ctx_with_token,
        *operands,
        sendcount=nitems_per_proc,
        sendtype=dtype_handle,
        recvcount=nitems_per_proc,
        recvtype=dtype_handle,
        comm=_np.int64(to_mpi_handle(comm)),
    )

    results = list(results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Platform-specific lowering rules (all use the same FFI implementation)
mpi_alltoall_xla_encode_cpu = translation_rule_cpu(_mpi_alltoall_xla_encode)
mpi_alltoall_xla_encode_cuda = translation_rule_cuda(_mpi_alltoall_xla_encode)
mpi_alltoall_xla_encode_xpu = translation_rule_xpu(_mpi_alltoall_xla_encode)


# This function evaluates only the shapes during AST construction
def mpi_alltoall_abstract_eval(xs, comm):
    return ShapedArray(xs.shape, xs.dtype), {ordered_effect}


mpi_alltoall_p.def_impl(mpi_alltoall_impl)
mpi_alltoall_p.def_effectful_abstract_eval(mpi_alltoall_abstract_eval)

# assign to the primitive the correct encoder
register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_cpu, platform="cpu")
register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_cuda, platform="cuda")
register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_xpu, platform="xpu")
