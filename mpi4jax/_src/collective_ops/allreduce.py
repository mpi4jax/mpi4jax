import numpy as _np
from mpi4py import MPI as _MPI

from jax import core
from jax.core import Primitive, Tracer, Token
from jax.interpreters import ad, batching
from jax.lax import create_token

from jax.interpreters import mlir
import jaxlib.mlir.ir as ir

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    unpack_hashable,
    wrap_as_hashable,
    as_mhlo_constant,
    get_default_layouts,
    effect,
    prefer_notoken,
)
from ..jax_compat import custom_call, token_type, ShapedArray
from ..decorators import (
    translation_rule_cpu,
    translation_rule_gpu,
    translation_rule_xpu,
)
from ..validation import enforce_types
from ..comm import get_default_comm

from ..xla_bridge.device_descriptors import build_allreduce_descriptor

# The Jax primitive
mpi_allreduce_p = Primitive("allreduce_mpi")  # Create the primitive
mpi_allreduce_impl = default_primitive_impl(mpi_allreduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def allreduce(x, op, *, comm=None, token=None):
    """Perform an allreduce operation.

    .. note::

       This primitive can be differentiated via :func:`jax.grad` and related functions
       if ``op`` is :obj:`mpi4py.MPI.SUM`.

    Arguments:
        x: Array or scalar input.
        op (mpi4py.MPI.Op): The reduction operator (e.g :obj:`mpi4py.MPI.SUM`).
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Result of the allreduce operation.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if prefer_notoken():
        from mpi4jax.experimental.notoken import allreduce

        return allreduce(x, op, comm=comm), token

    if comm is None:
        comm = get_default_comm()

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    return tuple(mpi_allreduce_p.bind(x, token, op=op, comm=comm, transpose=False))


# This function compiles the operation
# transpose is a boolean flag that signals whever this is the forward pass
# performing the MPI reduction, or the transposed pass, which is trivial
@translation_rule_cpu
def mpi_allreduce_xla_encode_cpu(ctx, x, token, op, comm, transpose):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    if transpose:
        assert op == _MPI.SUM
        return [x, token]

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(to_mpi_handle(op), _np.uintp),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(to_dtype_handle(x_nptype), _np.uintp),
        token,
    )

    return custom_call(
        b"mpi_allreduce",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    ).results


def mpi_allreduce_xla_encode_device(ctx, x, token, op, comm, transpose):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    if transpose:
        assert op == _MPI.SUM
        return [x, token]

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        *token_type(),
    ]

    operands = (
        x,
        token,
    )

    descriptor = build_allreduce_descriptor(
        _np.intc(nitems),
        to_mpi_handle(op),
        to_mpi_handle(comm),
        to_dtype_handle(x_nptype),
    )

    return custom_call(
        b"mpi_allreduce",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    ).results


mpi_allreduce_xla_encode_gpu = translation_rule_gpu(mpi_allreduce_xla_encode_device)
mpi_allreduce_xla_encode_xpu = translation_rule_xpu(mpi_allreduce_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_allreduce_abstract_eval(xs, token, op, comm, transpose):
    return (
        ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    ), {effect}


def mpi_allreduce_batch_eval(in_args, batch_axes, op, comm, transpose):
    x, token = in_args
    res = mpi_allreduce_p.bind(x, token, op=op, comm=comm, transpose=transpose)
    return res, batch_axes


def mpi_allreduce_value_and_jvp(in_args, tan_args, op, comm, transpose):
    x, token = in_args
    x_tan, token_tan = tan_args

    if unpack_hashable(op) != _MPI.SUM:
        raise NotImplementedError(
            "The adjoint of allreduce is only defined for op=MPI.SUM"
        )

    val, token = mpi_allreduce_p.bind(x, token, op=op, comm=comm, transpose=transpose)

    # throw away return token to work around jax#6285
    jvp, token_jvp = mpi_allreduce_p.bind(
        x_tan, token, op=op, comm=comm, transpose=transpose
    )
    return (val, token), (jvp, ad.Zero.from_value(token_jvp))


def mpi_allreduce_transpose_rule(tan_args, *x_args, op, comm, transpose):
    _, token = x_args
    x_tan, token_tan = tan_args

    if unpack_hashable(op) != _MPI.SUM:
        raise NotImplementedError(
            "The linear transpose of allreduce is only defined for op=MPI.SUM"
        )

    res, token = mpi_allreduce_p.bind(
        x_tan, token, op=op, comm=comm, transpose=(not transpose)
    )
    return res, token_tan


mpi_allreduce_p.multiple_results = True
mpi_allreduce_p.def_impl(mpi_allreduce_impl)
mpi_allreduce_p.def_effectful_abstract_eval(mpi_allreduce_abstract_eval)

batching.primitive_batchers[mpi_allreduce_p] = mpi_allreduce_batch_eval

ad.primitive_jvps[mpi_allreduce_p] = mpi_allreduce_value_and_jvp
ad.primitive_transposes[mpi_allreduce_p] = mpi_allreduce_transpose_rule

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_allreduce_p, mpi_allreduce_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_allreduce_p, mpi_allreduce_xla_encode_gpu, platform="cuda")
mlir.register_lowering(mpi_allreduce_p, mpi_allreduce_xla_encode_xpu, platform="xpu")
