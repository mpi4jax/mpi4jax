import numpy as _np
from mpi4py import MPI as _MPI

from jax import core
from jax.core import Primitive, Tracer, Token
from jax.interpreters import batching
from jax.lax import create_token

from jax.interpreters import mlir

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
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
mpi_barrier_p = Primitive("barrier_mpi")  # Create the primitive
mpi_barrier_impl = default_primitive_impl(mpi_barrier_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def barrier(*, comm=None, token=None):
    """Perform a barrier operation.

    Arguments:
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Token:
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token()

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)
    return mpi_barrier_p.bind(token, comm=comm)


# This function compiles the operation
# transpose is a boolean flag that signals whever this is the forward pass
# performing the MPI reduction, or the transposed pass, which is trivial
@translation_rule_cpu
def mpi_barrier_xla_encode_cpu(ctx, token, comm):
    comm = unpack_hashable(comm)

    out_types = token_type()

    operands = (
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        token,
    )

    # JAX insists on outputs being iterable
    return [
        hlo_custom_call(
            b"mpi_barrier",
            out_types=out_types,
            operands=operands,
            operand_layouts=get_default_layouts(operands),
            result_layouts=get_default_layouts(out_types),
            has_side_effect=True,
        )
    ]


@translation_rule_gpu
def mpi_barrier_xla_encode_gpu(ctx, token, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_barrier_descriptor

    comm = unpack_hashable(comm)

    out_types = token_type()

    operands = (token,)

    descriptor = build_barrier_descriptor(to_mpi_handle(comm))

    # JAX insists on outputs being iterable
    return [
        hlo_custom_call(
            b"mpi_barrier",
            out_types=out_types,
            operands=operands,
            operand_layouts=get_default_layouts(operands),
            result_layouts=get_default_layouts(out_types),
            has_side_effect=True,
            backend_config=descriptor,
        )
    ]


# This function evaluates only the shapes during AST construction
def mpi_barrier_abstract_eval(token, comm):
    return core.abstract_token, {effect}


def mpi_barrier_batch_eval(in_args, batch_axes, comm):
    token = in_args[0]
    res = mpi_barrier_p.bind(token, comm=comm)
    return res, batch_axes


mpi_barrier_p.def_impl(mpi_barrier_impl)
mpi_barrier_p.def_effectful_abstract_eval(mpi_barrier_abstract_eval)

batching.primitive_batchers[mpi_barrier_p] = mpi_barrier_batch_eval

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_barrier_p, mpi_barrier_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_barrier_p, mpi_barrier_xla_encode_gpu, platform="cuda")
