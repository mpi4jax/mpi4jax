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
mpi_gather_p = Primitive("gather_mpi")  # Create the primitive
mpi_gather_impl = default_primitive_impl(mpi_gather_p)


# This function applies the primitive to an AST
@enforce_types(
    root=(_np.integer),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def gather(
    x,
    root,
    *,
    comm=None,
    token=NOTSET,
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

    Returns:
        DeviceArray: Received data on root process, otherwise unmodified input.
    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    rank = comm.Get_rank()
    comm = wrap_as_hashable(comm)

    res = mpi_gather_p.bind(
        x,
        root=root,
        comm=comm,
    )

    if rank != root:
        return x

    return res


def _mpi_gather_xla_encode(ctx, x, root, comm):
    """Common lowering for all platforms using jax.ffi.ffi_lowering."""
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
        "mpi_gather_ffi",
        has_side_effect=True,
    )

    results = lowering_rule(
        ctx_with_token,
        *operands,
        sendcount=nitems,
        sendtype=dtype_handle,
        recvcount=nitems,
        recvtype=dtype_handle,
        root=_np.int64(root),
        comm=_np.int64(to_mpi_handle(comm)),
    )

    results = list(results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Platform-specific lowering rules (all use the same FFI implementation)
mpi_gather_xla_encode_cpu = translation_rule_cpu(_mpi_gather_xla_encode)
mpi_gather_xla_encode_cuda = translation_rule_cuda(_mpi_gather_xla_encode)
mpi_gather_xla_encode_xpu = translation_rule_xpu(_mpi_gather_xla_encode)


# This function evaluates only the shapes during AST construction
def mpi_gather_abstract_eval(x, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        out_shape = (size, *x.shape)
    else:
        out_shape = (0,)

    return ShapedArray(out_shape, x.dtype), {ordered_effect}


mpi_gather_p.def_impl(mpi_gather_impl)
mpi_gather_p.def_effectful_abstract_eval(mpi_gather_abstract_eval)

# assign to the primitive the correct encoder
register_lowering(mpi_gather_p, mpi_gather_xla_encode_cpu, platform="cpu")
register_lowering(mpi_gather_p, mpi_gather_xla_encode_cuda, platform="cuda")
register_lowering(mpi_gather_p, mpi_gather_xla_encode_xpu, platform="xpu")
