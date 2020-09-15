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
    default_primitive_impl,
)

from ..warn import warn_missing_omnistaging

# The Jax primitive
mpi_recv_p = Primitive("recv_mpi")  # Create the primitive
mpi_recv_impl = default_primitive_impl(mpi_recv_p)


# This function applies the primitive to an AST
def Recv(
    x,
    source=_MPI.ANY_SOURCE,
    tag=_MPI.ANY_TAG,
    comm=_MPI.COMM_WORLD,
    status=None,
    token=None,
):
    if token is None:
        token = create_token(x)

    comm = wrap_as_hashable(comm)
    return mpi_recv_p.bind(x, token, source=source, tag=tag, comm=comm, status=status)


#  This function compiles the operation
def mpi_recv_xla_encode(c, x, token, source, tag, comm, status):
    from ..cython.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    warn_missing_omnistaging()

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

    if status is None:
        _status = MPI_STATUS_IGNORE_ADDR
    else:
        _status = to_mpi_ptr(status)

    operands = (
        _nitems,
        _constant_s32_scalar(c, source),
        _constant_s32_scalar(c, tag),
        _constant_u64_scalar(c, to_mpi_ptr(comm)),
        _constant_u64_scalar(c, _dtype_ptr),
        _constant_u64_scalar(c, _status),
        token,
    )

    out = _ops.CustomCall(
        c, b"mpi_recv", operands=operands, shape=sh, has_side_effect=True,
    )

    return out


# This function evaluates only the shapes during AST construction
def mpi_recv_abstract_eval(xs, token, source, tag, comm, status):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        abstract_arrays.abstract_token,
    )


mpi_recv_p.multiple_results = True
mpi_recv_p.def_impl(mpi_recv_impl)
mpi_recv_p.def_abstract_eval(mpi_recv_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_recv_p] = mpi_recv_xla_encode
