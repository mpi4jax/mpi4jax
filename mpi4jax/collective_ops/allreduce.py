import numpy as _np

from mpi4py import MPI as _MPI

from jax import abstract_arrays
from jax.lax import create_token
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla

from ..utils import (
    to_mpi_ptr,
    _unpack_builder,
    _ops,
    _constant_s32_scalar,
    _constant_u64_scalar,
    dtype_ptr,
    wrap_as_hashable,
    unpack_hashable,
)

from ..warn import warn_missing_omnistaging

# The Jax primitive
mpi_allreduce_p = Primitive("allreduce_mpi")  # Create the primitive


# This function applies the primitive to an AST
def Allreduce(x, op, comm=_MPI.COMM_WORLD, token=None):
    if token is None:
        token = create_token(x)

    return mpi_allreduce_p.bind(x, token, op=op, comm=comm)


#  this function executes the primitive, when not under any transformation
def mpi_allreduce_impl(x, token, op, comm):
    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    return xla.apply_primitive(mpi_allreduce_p, x, token, op=op, comm=comm)


#  This function compiles the operation
def mpi_allreduce_xla_encode(c, x, token, op, comm):
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


# This function evaluates only the shapes during AST construction
def mpi_allreduce_abstract_eval(xs, token, op, comm):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        abstract_arrays.abstract_token,
    )


mpi_allreduce_p.multiple_results = True
mpi_allreduce_p.def_impl(mpi_allreduce_impl)
mpi_allreduce_p.def_abstract_eval(mpi_allreduce_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_allreduce_p] = mpi_allreduce_xla_encode
