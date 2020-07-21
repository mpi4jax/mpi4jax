import numpy as _np
import ctypes

from mpi4py import MPI as _MPI

from jax import abstract_arrays
from jax import numpy as jnp
from jax.core import Primitive
from jax.lib import xla_client
from jax.interpreters import xla, ad, batching

from ..utils import (
    to_mpi_ptr,
    MPIComm_from_ptr,
    MPIOp_from_ptr,
    _unpack_builder,
    _ops,
    _constant_s32_scalar,
    _constant_u64_scalar,
    dtype_ptr,
)

# The Jax primitive
mpi_allreduce_p = Primitive("sum_inplace_mpi")  # Create the primitive

# This function applies the primitive to an AST
def Allreduce(x, op, comm=_MPI.COMM_WORLD):
    op_ptr = to_mpi_ptr(op)
    comm_ptr = to_mpi_ptr(comm)

    return mpi_allreduce_p.bind(x, op=op_ptr, comm=comm_ptr)


#  this function executes the primitive, when not under any transformation
def mpi_allreduce_impl(x, op, comm):
    # TODO: make this support gpus (use cupy?)
    inpt = _np.asarray(x)
    out = _np.zeros_like(x)

    # rebuild comm and op
    _op = MPIOp_from_ptr(op)
    _comm = MPIComm_from_ptr(comm)

    _comm.Allreduce(inpt, out, op=_op)

    res = jnp.array(out, dtype=x.dtype)

    # put the result on the correct device if needed
    if not (res.device_buffer.device() == x.device_buffer.device()):
        res = jax.device_put(res, device=x.device_buffer.device())

    return res


#  This function compiles the operation
def mpi_allreduce_xla_encode(c, x, op, comm):
    c = _unpack_builder(c)
    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = dims[0]
    for el in dims[1:]:
        nitems *= el

    _nitems = _constant_s32_scalar(c, nitems)

    _dtype_ptr = dtype_ptr(dtype)

    sh = xla_client.Shape.array_shape(dtype, dims)

    return _ops.CustomCall(
        c,
        b"mpi_allreduce",
        operands=(
            _nitems,
            x,
            _constant_u64_scalar(c, op),
            _constant_u64_scalar(c, comm),
            _constant_u64_scalar(c, _dtype_ptr),
        ),
        shape=sh,
    )


# This function evaluates only the shapes during AST construction
def mpi_allreduce_abstract_eval(xs, op, comm):
    return abstract_arrays.ShapedArray(xs.shape, xs.dtype)


# This function binds the batched transformation.
def mpi_allreduce_batching(in_args, batch_axes, **kwargs):
    (x,) = in_args
    res = mpi_allreduce_p.bind(x, **kwargs)
    return res, batch_axes[0]


def mpi_allreduce_value_and_jvp(in_args, tan_args, **kwargs):
    (x,) = in_args
    (x_tan,) = tan_args
    res = mpi_allreduce_p.bind(x, **kwargs)
    jvp = x_tan
    return (res, jvp)


mpi_allreduce_p.def_impl(mpi_allreduce_impl)
mpi_allreduce_p.def_abstract_eval(mpi_allreduce_abstract_eval)

batching.primitive_batchers[mpi_allreduce_p] = mpi_allreduce_batching
ad.primitive_jvps[mpi_allreduce_p] = mpi_allreduce_value_and_jvp

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_allreduce_p] = mpi_allreduce_xla_encode
