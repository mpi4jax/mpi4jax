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
from ..warn import warn_missing_omnistaging

# The Jax primitive
mpi_allreduce_p = Primitive("allreduce_mpi")  # Create the primitive
mpi_allreduce_impl = default_primitive_impl(mpi_allreduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def Allreduce(x, op, comm=_MPI.COMM_WORLD, token=None, _transpose=False):
    """
    Performs the Allreduce operation `op` on the input `x` using the
    communicator `comm` which defaults to the world comunicator.
    An optional token can be passed, which is used to force jax to execute
    MPI operations in the correct order.

    Arguments:
        x: Array or scalar input.
        op: The reduction operation `MPI.Op` (e.g: MPI.SUM)
        comm: The communicator (defaults to MPI.COMM_WORLD)
        token: token to force a sequential order in the operations (default=None)

    Returns:
        res: result of the allreduce operation
        new_token: a new, modified token, that depends on this operation.
            This result can be ignored if result forces a data dependency.
    """

    # The extra argument _transpose is an implementation detail. It is used to
    # keep track of whever we are computing the forward or transposition of
    # Allreduce, because we are 'cheating' and not performing MPI operations
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
    warn_missing_omnistaging()

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
                "The linear transpose of Allreduce for {} is not defined".format(op)
            )

        return _ops.Tuple(c, [x, token])
    else:
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
    from mpi4jax.cython import HAS_GPU_EXT

    if not HAS_GPU_EXT:
        raise RuntimeError(
            "mpi4jax GPU extensions failed to build, "
            "so it cannot be used in GPU contexts"
        )

    from mpi4jax.cython.mpi_xla_bridge_gpu import build_allreduce_descriptor

    warn_missing_omnistaging()

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
                "The linear transpose of Allreduce for {} is not defined".format(op)
            )

        return _ops.Tuple(c, [x, token])
    else:
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
