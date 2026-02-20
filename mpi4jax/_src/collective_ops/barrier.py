import numpy as _np
from mpi4py import MPI as _MPI

from jax.ffi import ffi_lowering
from jax.interpreters import batching

from mpi4jax._src.utils import (
    HashableMPIType,
    default_primitive_impl,
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
mpi_barrier_p = Primitive("barrier_mpi")  # Create the primitive
mpi_barrier_impl = default_primitive_impl(mpi_barrier_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def barrier(*, comm=None, token=NOTSET):
    """Perform a barrier operation.

    Arguments:
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)
    return mpi_barrier_p.bind(comm=comm)


def _mpi_barrier_xla_encode(ctx, comm):
    """Common lowering for all platforms using jax.ffi.ffi_lowering."""
    comm = unpack_hashable(comm)

    token = get_token_effect(ctx, ordered_effect)

    # barrier has no buffer outputs, but we need token for ordering
    out_types = [token_type()]
    operands = (token,)

    lowering_rule = ffi_lowering(
        "mpi_barrier_ffi",
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        result_types=out_types,
        has_side_effect=True,
        skip_ffi_layout_processing=True,
    )

    results = lowering_rule(
        ctx,
        *operands,
        comm=_np.int64(to_mpi_handle(comm)),
    )

    results = list(results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Platform-specific lowering rules (all use the same FFI implementation)
mpi_barrier_xla_encode_cpu = translation_rule_cpu(_mpi_barrier_xla_encode)
mpi_barrier_xla_encode_cuda = translation_rule_cuda(_mpi_barrier_xla_encode)
mpi_barrier_xla_encode_xpu = translation_rule_xpu(_mpi_barrier_xla_encode)


# This function evaluates only the shapes during AST construction
def mpi_barrier_abstract_eval(comm):
    return (), {ordered_effect}


def mpi_barrier_batch_eval(in_args, batch_axes, comm):
    res = mpi_barrier_p.bind(comm=comm)
    return res, batch_axes


mpi_barrier_p.multiple_results = True
mpi_barrier_p.def_impl(mpi_barrier_impl)
mpi_barrier_p.def_effectful_abstract_eval(mpi_barrier_abstract_eval)

batching.primitive_batchers[mpi_barrier_p] = mpi_barrier_batch_eval

# assign to the primitive the correct encoder
register_lowering(mpi_barrier_p, mpi_barrier_xla_encode_cpu, platform="cpu")
register_lowering(mpi_barrier_p, mpi_barrier_xla_encode_cuda, platform="cuda")
register_lowering(mpi_barrier_p, mpi_barrier_xla_encode_xpu, platform="xpu")
