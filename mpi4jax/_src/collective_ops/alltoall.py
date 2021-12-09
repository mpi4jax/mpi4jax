import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive
from jax.interpreters import xla
from jax.lax import create_token
from jax.lib import xla_client

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    unpack_hashable,
    wrap_as_hashable,
    xla_constant_intc,
    xla_constant_uintptr,
)
from ..decorators import translation_rule_cpu, translation_rule_gpu
from ..validation import enforce_types
from ..comm import get_default_comm
from ..jax_compat import Tracer, Token


# The Jax primitive
mpi_alltoall_p = Primitive("alltoall_mpi")  # Create the primitive
mpi_alltoall_impl = default_primitive_impl(mpi_alltoall_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def alltoall(
    x,
    *,
    comm=None,
    token=None,
):
    """Perform an alltoall operation.

    Arguments:
        x: Array input to send. First axis must have size ``nproc``.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if comm is None:
        comm = get_default_comm()

    size = comm.Get_size()
    if x.shape[0] != size:
        raise ValueError("Alltoall input must have shape (nproc, ...)")

    comm = wrap_as_hashable(comm)

    return tuple(
        mpi_alltoall_p.bind(
            x,
            token,
            comm=comm,
        )
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_alltoall_xla_encode_cpu(c, x, token, comm):
    comm = unpack_hashable(comm)

    shape = c.GetShape(x)
    dtype = shape.element_type()
    dims = shape.dimensions()

    # compute total number of elements in array
    size = comm.Get_size()
    assert dims[0] == size
    nitems_per_proc = _np.prod(dims[1:], dtype=int)
    dtype_handle = to_dtype_handle(dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(dtype, dims),
            xla_client.Shape.token_shape(),
        ]
    )

    operands = (
        xla_constant_intc(c, nitems_per_proc),
        x,
        xla_constant_uintptr(c, dtype_handle),
        # we only support matching input and output arrays
        xla_constant_intc(c, nitems_per_proc),
        xla_constant_uintptr(c, dtype_handle),
        #
        xla_constant_uintptr(c, to_mpi_handle(comm)),
        token,
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_alltoall",
        operands=operands,
        shape=sh,
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_alltoall_xla_encode_gpu(c, x, token, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_alltoall_descriptor

    comm = unpack_hashable(comm)

    shape = c.GetShape(x)
    dtype = shape.element_type()
    dims = shape.dimensions()

    # compute total number of elements in send array
    size = comm.Get_size()
    assert dims[0] == size
    nitems_per_proc = _np.prod(dims[1:], dtype=int)
    dtype_handle = to_dtype_handle(dtype)

    sh = xla_client.Shape.tuple_shape(
        [
            xla_client.Shape.array_shape(dtype, dims),
            xla_client.Shape.token_shape(),
        ]
    )

    descriptor = build_alltoall_descriptor(
        nitems_per_proc,
        dtype_handle,
        # we only support matching input and output arrays
        nitems_per_proc,
        dtype_handle,
        #
        to_mpi_handle(comm),
    )

    return xla_client.ops.CustomCall(
        c,
        b"mpi_alltoall",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_alltoall_abstract_eval(xs, token, comm):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    )


mpi_alltoall_p.multiple_results = True
mpi_alltoall_p.def_impl(mpi_alltoall_impl)
mpi_alltoall_p.def_abstract_eval(mpi_alltoall_abstract_eval)

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_alltoall_p] = mpi_alltoall_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_alltoall_p] = mpi_alltoall_xla_encode_gpu
