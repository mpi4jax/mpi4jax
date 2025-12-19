import numpy as _np
from mpi4py import MPI as _MPI

from jax.interpreters import ad, batching
from jax.core import ShapedArray


import jaxlib.mlir.ir as ir
from jax._src.interpreters.mlir import custom_call

from mpi4jax._src.utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
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

from mpi4jax._src.xla_bridge.device_descriptors import build_allreduce_descriptor

# The Jax primitive
mpi_allreduce_p = Primitive("allreduce_mpi")  # Create the primitive
mpi_allreduce_impl = default_primitive_impl(mpi_allreduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def allreduce(x, op, *, comm=None, token=NOTSET):
    """Perform an allreduce operation.

    .. note::

       This primitive can be differentiated via :func:`jax.grad` and related functions
       if ``op`` is :obj:`mpi4py.MPI.SUM`.

    Arguments:
        x: Array or scalar input.
        op (mpi4py.MPI.Op): The reduction operator (e.g :obj:`mpi4py.MPI.SUM`).
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Result of the allreduce operation.

    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    return mpi_allreduce_p.bind(x, op=op, comm=comm, transpose=False)


# This function compiles the operation
# transpose is a boolean flag that signals whever this is the forward pass
# performing the MPI reduction, or the transposed pass, which is trivial
@translation_rule_cpu
def mpi_allreduce_xla_encode_cpu(ctx, x, op, comm, transpose):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    if transpose:
        assert op == _MPI.SUM
        return [x]

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        token_type(),
    ]

    token = get_token_effect(ctx, ordered_effect)

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(to_mpi_handle(op), _np.uintp),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(to_dtype_handle(x_nptype), _np.uintp),
        token,
    )

    result_obj = custom_call(
        b"mpi_allreduce",
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


def mpi_allreduce_xla_encode_device(ctx, x, op, comm, transpose):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    if transpose:
        assert op == _MPI.SUM
        return [x]

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        token_type(),
    ]

    token = get_token_effect(ctx, ordered_effect)

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

    result_obj = custom_call(
        b"mpi_allreduce",
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


mpi_allreduce_xla_encode_xpu = translation_rule_xpu(mpi_allreduce_xla_encode_device)
mpi_allreduce_xla_encode_cuda = translation_rule_cuda(mpi_allreduce_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_allreduce_abstract_eval(xs, op, comm, transpose):
    if not transpose:
        return ShapedArray(xs.shape, xs.dtype), {ordered_effect}
    else:
        # The transposition of an allreduce is just the identity, so it can be reordered
        # and does not come with an ordered effect.
        return ShapedArray(xs.shape, xs.dtype), {}


def mpi_allreduce_batch_eval(in_args, batch_axes, op, comm, transpose):
    (ax, *_) = batch_axes
    res = mpi_allreduce_p.bind(*in_args, op=op, comm=comm, transpose=transpose)
    return res, ax


def mpi_allreduce_value_and_jvp(in_args, tan_args, op, comm, transpose):
    (x,) = in_args
    (x_tan, *_) = tan_args

    if unpack_hashable(op) != _MPI.SUM:
        raise NotImplementedError(
            "The adjoint of allreduce is only defined for op=MPI.SUM"
        )

    val = mpi_allreduce_p.bind(x, op=op, comm=comm, transpose=transpose)
    jvp = mpi_allreduce_p.bind(x_tan, op=op, comm=comm, transpose=transpose)
    return val, jvp


def mpi_allreduce_transpose_rule(x_tan, *x_args, op, comm, transpose):
    if unpack_hashable(op) != _MPI.SUM:
        raise NotImplementedError(
            "The linear transpose of allreduce is only defined for op=MPI.SUM"
        )
    x_tan = ad.instantiate_zeros(x_tan)
    res = mpi_allreduce_p.bind(x_tan, op=op, comm=comm, transpose=(not transpose))
    return (res,)


mpi_allreduce_p.def_impl(mpi_allreduce_impl)
mpi_allreduce_p.def_effectful_abstract_eval(mpi_allreduce_abstract_eval)

batching.primitive_batchers[mpi_allreduce_p] = mpi_allreduce_batch_eval

ad.primitive_jvps[mpi_allreduce_p] = mpi_allreduce_value_and_jvp
ad.primitive_transposes[mpi_allreduce_p] = mpi_allreduce_transpose_rule

# assign to the primitive the correct encoder
register_lowering(mpi_allreduce_p, mpi_allreduce_xla_encode_cpu, platform="cpu")
register_lowering(mpi_allreduce_p, mpi_allreduce_xla_encode_cuda, platform="cuda")
register_lowering(mpi_allreduce_p, mpi_allreduce_xla_encode_xpu, platform="xpu")
