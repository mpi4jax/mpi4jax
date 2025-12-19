import numpy as _np
from mpi4py import MPI as _MPI

from jax.interpreters import batching
from jax._src.interpreters.mlir import custom_call

from mpi4jax._src.utils import (
    HashableMPIType,
    default_primitive_impl,
    to_mpi_handle,
    unpack_hashable,
    wrap_as_hashable,
    as_mhlo_constant,
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
from mpi4jax._src.comm import get_default_comm

from mpi4jax._src.xla_bridge.device_descriptors import build_barrier_descriptor


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


# This function compiles the operation
# transpose is a boolean flag that signals whever this is the forward pass
# performing the MPI reduction, or the transposed pass, which is trivial
@translation_rule_cpu
def mpi_barrier_xla_encode_cpu(ctx, comm):
    comm = unpack_hashable(comm)

    out_types = [token_type()]

    token = get_token_effect(ctx, ordered_effect)

    operands = (
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        token,
    )

    result_obj = custom_call(
        b"mpi_barrier",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


def mpi_barrier_xla_encode_device(ctx, comm):
    comm = unpack_hashable(comm)

    out_types = [token_type()]

    token = get_token_effect(ctx, ordered_effect)

    operands = (token,)

    descriptor = build_barrier_descriptor(to_mpi_handle(comm))

    result_obj = custom_call(
        b"mpi_barrier",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


mpi_barrier_xla_encode_xpu = translation_rule_xpu(mpi_barrier_xla_encode_device)
mpi_barrier_xla_encode_cuda = translation_rule_cuda(mpi_barrier_xla_encode_device)


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
