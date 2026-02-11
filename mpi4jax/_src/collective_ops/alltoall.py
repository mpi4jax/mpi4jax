import numpy as _np
from mpi4py import MPI as _MPI

import jaxlib.mlir.ir as ir
from jax._src.interpreters.mlir import custom_call
from jax.core import ShapedArray

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

from mpi4jax._src.xla_bridge.device_descriptors import build_alltoall_descriptor


# Check if FFI-based C++ implementation is available
def _has_ffi_support():
    try:
        from mpi4jax._src.xla_bridge import HAS_CPP_EXT, HAS_FFI_TARGETS

        return HAS_CPP_EXT and HAS_FFI_TARGETS
    except ImportError:
        return False


# The Jax primitive
mpi_alltoall_p = Primitive("alltoall_mpi")  # Create the primitive
mpi_alltoall_impl = default_primitive_impl(mpi_alltoall_p)


# This function applies the primitive to an AST
@enforce_types(
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
)
def alltoall(
    x,
    *,
    comm=None,
    token=NOTSET,
):
    """Perform an alltoall operation.

    Arguments:
        x: Array input to send. First axis must have size ``nproc``.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        DeviceArray: Received data.

    """
    raise_if_token_is_set(token)

    if comm is None:
        comm = get_default_comm()

    size = comm.Get_size()
    if x.shape[0] != size:
        raise ValueError("Alltoall input must have shape (nproc, ...)")

    comm = wrap_as_hashable(comm)

    return mpi_alltoall_p.bind(
        x,
        comm=comm,
    )


# FFI-based CPU lowering rule using jax.ffi (new typed API)
def mpi_alltoall_xla_encode_cpu_ffi(ctx, x, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    size = comm.Get_size()
    assert dims[0] == size
    nitems_per_proc = int(_np.prod(dims[1:], dtype=int))
    dtype_handle = int(to_dtype_handle(x_nptype))

    token = get_token_effect(ctx, ordered_effect)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        token_type(),
    ]

    operands = (x, token)

    backend_config = {
        "sendcount": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), nitems_per_proc
        ),
        "sendtype": ir.IntegerAttr.get(ir.IntegerType.get_unsigned(64), dtype_handle),
        "recvcount": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), nitems_per_proc
        ),
        "recvtype": ir.IntegerAttr.get(ir.IntegerType.get_unsigned(64), dtype_handle),
        "comm": ir.IntegerAttr.get(
            ir.IntegerType.get_unsigned(64), int(to_mpi_handle(comm))
        ),
    }

    result_obj = custom_call(
        b"mpi_alltoall_ffi",
        result_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
        api_version=4,
        backend_config=backend_config,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Legacy CPU lowering rule (api_version=0)
def mpi_alltoall_xla_encode_cpu_legacy(ctx, x, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    size = comm.Get_size()
    assert dims[0] == size
    nitems_per_proc = _np.prod(dims[1:], dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        token_type(),
    ]

    token = get_token_effect(ctx, ordered_effect)

    operands = (
        as_mhlo_constant(nitems_per_proc, _np.intc),
        x,
        as_mhlo_constant(dtype_handle, _np.uintp),
        # we only support matching input and output arrays
        as_mhlo_constant(nitems_per_proc, _np.intc),
        as_mhlo_constant(dtype_handle, _np.uintp),
        #
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        token,
    )

    result_obj = custom_call(
        b"mpi_alltoall",
        result_types=out_types,
        operands=operands,
        # force c order because first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)

    return results


# Choose which CPU lowering to use based on FFI availability
@translation_rule_cpu
def mpi_alltoall_xla_encode_cpu(ctx, x, comm):
    import os

    use_ffi = os.getenv("MPI4JAX_USE_FFI", "true").lower() in ("true", "1", "on")

    if use_ffi and _has_ffi_support():
        return mpi_alltoall_xla_encode_cpu_ffi(ctx, x, comm)
    else:
        return mpi_alltoall_xla_encode_cpu_legacy(ctx, x, comm)


def mpi_alltoall_xla_encode_device(ctx, x, comm):
    comm = unpack_hashable(comm)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    size = comm.Get_size()
    assert dims[0] == size
    nitems_per_proc = _np.prod(dims[1:], dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        token_type(),
    ]

    token = get_token_effect(ctx, ordered_effect)

    operands = (
        x,
        token,
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

    result_obj = custom_call(
        b"mpi_alltoall",
        result_types=out_types,
        operands=operands,
        # force c order because first axis is special
        operand_layouts=get_default_layouts(operands, order="c"),
        result_layouts=get_default_layouts(out_types, order="c"),
        has_side_effect=True,
        backend_config=descriptor,
    )

    results = list(result_obj.results)
    token = results.pop(-1)
    set_token_effect(ctx, ordered_effect, token)
    return results


mpi_alltoall_xla_encode_xpu = translation_rule_xpu(mpi_alltoall_xla_encode_device)
mpi_alltoall_xla_encode_cuda = translation_rule_cuda(mpi_alltoall_xla_encode_device)


# This function evaluates only the shapes during AST construction
def mpi_alltoall_abstract_eval(xs, comm):
    return ShapedArray(xs.shape, xs.dtype), {ordered_effect}


mpi_alltoall_p.def_impl(mpi_alltoall_impl)
mpi_alltoall_p.def_effectful_abstract_eval(mpi_alltoall_abstract_eval)

# assign to the primitive the correct encoder
register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_cpu, platform="cpu")
register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_cuda, platform="cuda")
register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_xpu, platform="xpu")
