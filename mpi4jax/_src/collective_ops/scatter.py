import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive
from jax.interpreters import xla
from jax.lax import create_token
from jax.lib import xla_client

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
mpi_scatter_p = Primitive("scatter_mpi")  # Create the primitive
mpi_scatter_impl = default_primitive_impl(mpi_scatter_p)


# This function applies the primitive to an AST
@enforce_types(
    root=(_np.integer),
    comm=(_MPI.Intracomm, HashableMPIType),
    token=(type(None), xla.Token, core.Tracer),
)
def scatter(
    x,
    root,
    comm=_MPI.COMM_WORLD,
    token=None,
):
    """Perform a scatter operation.

    .. warning::

        Unlike mpi4py's scatter, this returns a *new* array with the received data.

    Arguments:
        x: Array or scalar input with the correct shape and dtype. On the root process,
           this contains the data to send, and its first axis must have size ``nproc``.
           On non-root processes, this may contain arbitrary data and will not be
           overwritten.
        root (int): Rank of the root MPI process.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            :obj:`COMM_WORLD`).
        token: XLA token to use to ensure correct execution order. If not given,
            a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.
    """
    if token is None:
        token = create_token(x)

    rank = comm.Get_rank()
    if rank == root:
        size = comm.Get_size()
        if x.shape[0] != size:
            raise ValueError("Scatter input must have shape (nproc, ...)")

    comm = wrap_as_hashable(comm)

    return mpi_scatter_p.bind(
        x,
        token,
        root=root,
        comm=comm,
    )


#  This function compiles the operation
def mpi_scatter_xla_encode_cpu(c, x, token, root, comm):
    comm = unpack_hashable(comm)

    c = _unpack_builder(c)

    shape = c.GetShape(x)
    dtype = shape.element_type()
    dims = shape.dimensions()

    rank = comm.Get_rank()
    if rank == root:
        dims = dims[1:]

    # compute total number of elements in array
    _nitems = _constant_s32_scalar(c, _np.prod(dims, dtype=int))
    _dtype_ptr = dtype_ptr(dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(dtype, dims),
            xla_client.Shape.token_shape(),
        ]
    )

    operands = (
        _nitems,
        x,
        _constant_u64_scalar(c, _dtype_ptr),
        # we only support matching input and output arrays
        _nitems,
        _constant_u64_scalar(c, _dtype_ptr),
        #
        _constant_s32_scalar(c, root),
        _constant_u64_scalar(c, to_mpi_ptr(comm)),
        token,
    )

    return _ops.CustomCall(
        c,
        b"mpi_scatter",
        operands=operands,
        shape=sh,
        has_side_effect=True,
    )


def mpi_scatter_xla_encode_gpu(c, x, token, root, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_scatter_descriptor

    comm = unpack_hashable(comm)

    c = _unpack_builder(c)

    shape = c.GetShape(x)
    dtype = shape.element_type()
    dims = shape.dimensions()

    rank = comm.Get_rank()
    if rank == root:
        dims = dims[1:]

    # compute total number of elements in array
    _nitems = _np.prod(dims, dtype=int)
    _dtype_ptr = dtype_ptr(dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(dtype, dims),
            xla_client.Shape.token_shape(),
        ]
    )

    descriptor = build_scatter_descriptor(
        _nitems,
        _dtype_ptr,
        # we only support matching input and output arrays
        _nitems,
        _dtype_ptr,
        #
        root,
        to_mpi_ptr(comm),
    )

    return _ops.CustomCall(
        c,
        b"mpi_scatter",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_scatter_abstract_eval(x, token, root, comm):
    comm = unpack_hashable(comm)
    rank = comm.Get_rank()
    if rank == root:
        out_shape = x.shape[1:]
    else:
        out_shape = x.shape

    return (
        abstract_arrays.ShapedArray(out_shape, x.dtype),
        abstract_arrays.abstract_token,
    )


mpi_scatter_p.multiple_results = True
mpi_scatter_p.def_impl(mpi_scatter_impl)
mpi_scatter_p.def_abstract_eval(mpi_scatter_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_scatter_p] = mpi_scatter_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_scatter_p] = mpi_scatter_xla_encode_gpu
