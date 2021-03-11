import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive
from jax.interpreters import ad, xla
from jax.lib import xla_client

from jax.lax import create_token
from ..utils import (
    HashableMPIType,
    _constant_s32_scalar,
    _constant_u64_scalar,
    _ops,
    _unpack_builder,
    default_primitive_impl,
    dtype_ptr,
    to_mpi_ptr,
    unpack_hashable,
    wrap_as_hashable,
)
from ..validation import enforce_types

# The Jax primitive
mpi_allreduce_p = Primitive("allreduce_mpi")  # Create the primitive
mpi_allreduce_impl = default_primitive_impl(mpi_allreduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def allreduce(x, op, comm=_MPI.COMM_WORLD, token=None, _transpose=False):
    """Perform an allreduce operation.

    .. note::

       This primitive can be differentiated via :func:`jax.grad` and related functions
       if ``op`` is :obj:`mpi4py.MPI.SUM`.

    Arguments:
        x: Array or scalar input.
        op (mpi4py.MPI.Op): The reduction operator (e.g :obj:`mpi4py.MPI.SUM`).
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            :obj:`COMM_WORLD`).
        token: XLA token to use to ensure correct execution order. If not given,
            a new token is generated.
        _transpose (bool): Used internally.

    Returns:
        Tuple[DeviceArray, Token]:
            - Result of the allreduce operation.
            - A new, modified token, that depends on this operation.

    """

    # The extra argument _transpose is an implementation detail. It is used to
    # keep track of whever we are computing the forward or transposition of
    # allreduce, because we are 'cheating' and not performing MPI operations
    # on the tranposed (even though, in principle, we should) in order
    # to be more efficient.

    if token is None:
        token = create_token(x)

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    return mpi_allreduce_p.bind(x, token, op=op, comm=comm, transpose=_transpose)


# This function compiles the operation
# transpose is a boolean flag that signals whever this is the forward pass
# performing the MPI reduction, or the transposed pass, which is trivial
def mpi_allreduce_xla_encode_cpu(c, x, token, op, comm, transpose):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    _nitems = _constant_s32_scalar(c, _np.prod(dims, dtype=int))

    _dtype_ptr = dtype_ptr(dtype)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    if transpose:
        if op != _MPI.SUM:
            raise NotImplementedError(
                "The linear transpose of allreduce for {} is not defined".format(op)
            )

        return _ops.Tuple(c, [x, token])

    return _ops.CustomCall(
        c,
        b"mpi_allreduce",
        operands=(
            _nitems,
            x,
            _constant_u64_scalar(c, to_mpi_ptr(op)),
            _constant_u64_scalar(c, to_mpi_ptr(comm)),
            _constant_u64_scalar(c, _dtype_ptr),
            token,
        ),
        shape=sh,
        has_side_effect=True,
    )


def mpi_allreduce_xla_encode_gpu(c, x, token, op, comm, transpose):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_allreduce_descriptor

    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    _nitems = _np.prod(dims, dtype=int)

    _dtype_ptr = dtype_ptr(dtype)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    descriptor = build_allreduce_descriptor(
        _nitems,
        to_mpi_ptr(op),
        to_mpi_ptr(comm),
        _dtype_ptr,
    )

    if transpose:
        if op != _MPI.SUM:
            raise NotImplementedError(
                "The linear transpose of allreduce for {} is not defined".format(op)
            )

        return _ops.Tuple(c, [x, token])

    return _ops.CustomCall(
        c,
        b"mpi_allreduce",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_allreduce_abstract_eval(xs, token, op, comm, transpose):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        abstract_arrays.abstract_token,
    )


def mpi_allreduce_value_and_jvp(in_args, tan_args, op, comm, transpose):
    x, token = in_args
    x_tan, token_tan = tan_args

    res = mpi_allreduce_p.bind(x, token, op=op, comm=comm, transpose=transpose)

    # Identify the correct adjoint
    op_ = unpack_hashable(op)

    if op_ == _MPI.SUM:
        jvp = mpi_allreduce_p.bind(x_tan, token, op=op, comm=comm, transpose=transpose)
    else:
        raise NotImplementedError(
            "The adjoint of allreduce for {} operation is not defined".format(op_)
        )

    return (res, jvp)


def mpi_allreduce_transpose_rule(tan_args, *x_args, op, comm, transpose):
    _, token = x_args
    t, _ = tan_args

    # Identify the correct adjoint
    op_ = unpack_hashable(op)

    if op_ == _MPI.SUM:
        return mpi_allreduce_p.bind(
            t, token, op=op, comm=comm, transpose=(not transpose)
        )
    else:
        raise NotImplementedError(
            "The linear transpose of allreduce for {} is not defined".format(op_)
        )


mpi_allreduce_p.multiple_results = True
mpi_allreduce_p.def_impl(mpi_allreduce_impl)
mpi_allreduce_p.def_abstract_eval(mpi_allreduce_abstract_eval)

ad.primitive_jvps[mpi_allreduce_p] = mpi_allreduce_value_and_jvp
ad.primitive_transposes[mpi_allreduce_p] = mpi_allreduce_transpose_rule

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_allreduce_p] = mpi_allreduce_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_allreduce_p] = mpi_allreduce_xla_encode_gpu
