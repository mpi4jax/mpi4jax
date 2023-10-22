import os
import warnings
import functools

# global variables to keep track of state
_cuda_mpi_setup_done = False
_hip_mpi_setup_done = False


def ensure_gpu_ext():
    from .xla_bridge import HAS_GPU_CUDA_EXT, HAS_GPU_HIP_EXT

    if not HAS_GPU_CUDA_EXT and not HAS_GPU_HIP_EXT:
        raise ImportError(
            "The mpi4jax GPU extensions could not be imported. "
            "Please re-build mpi4jax with CUDA or HIP support and try again."
        )


def _is_truthy(str_val):
    return str_val.lower() in ("true", "1", "on")


def _is_falsy(str_val):
    return str_val.lower() in ("false", "0", "off")


def setup_cuda_mpi():
    global _cuda_mpi_setup_done

    if _cuda_mpi_setup_done:
        return

    _cuda_mpi_setup_done = True

    gpu_copy_behavior = os.getenv("MPI4JAX_USE_CUDA_MPI", "")

    if _is_truthy(gpu_copy_behavior):
        has_cuda_mpi = True
    elif _is_falsy(gpu_copy_behavior):
        has_cuda_mpi = False
    else:
        has_cuda_mpi = False
        warn_msg = (
            "Not using CUDA-enabled MPI. "
            "If you are sure that your MPI library is built with CUDA support, "
            "set MPI4JAX_USE_CUDA_MPI=1. To silence this warning, "
            "set MPI4JAX_USE_CUDA_MPI=0."
        )
        warnings.warn(warn_msg)

    try:
        from .xla_bridge import mpi_xla_bridge_gpu_cuda

        mpi_xla_bridge_gpu_cuda.set_copy_to_host(not has_cuda_mpi)
    except ImportError:
        warnings.warn("CUDA xla_bridge is not compiled")


def setup_hip_mpi():
    global _hip_mpi_setup_done

    if _hip_mpi_setup_done:
        return

    _hip_mpi_setup_done = True

    gpu_copy_behavior = os.getenv("MPI4JAX_USE_HIP_MPI", "")

    if _is_truthy(gpu_copy_behavior):
        has_hip_mpi = True
    elif _is_falsy(gpu_copy_behavior):
        has_hip_mpi = False
    else:
        has_hip_mpi = False
        warn_msg = (
            "Not using HIP-enabled MPI. "
            "If you are sure that your MPI library is built with HIP support, "
            "set MPI4JAX_USE_HIP_MPI=1. To silence this warning, "
            "set MPI4JAX_USE_HIP_MPI=0."
        )
        warnings.warn(warn_msg)

    try:
        from .xla_bridge import mpi_xla_bridge_gpu_hip

        mpi_xla_bridge_gpu_hip.set_copy_to_host(not has_hip_mpi)
    except ImportError:
        warnings.warn("HIP xla_bridge is not compiled")


def translation_rule_cpu(func):
    """XLA primitive translation rule on CPU for mpi4jax custom calls.

    This runs generic setup and boilerplate functions.
    """
    # NOTE: currently does nothing, but we keep it for consistency

    # functions to call before running the translation rule
    setup_funcs = ()

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for f in setup_funcs:
            f()
        return func(*args, **kwargs)

    return wrapped


def translation_rule_gpu(func):
    """XLA primitive translation rule on GPU for mpi4jax custom calls.

    This runs generic setup and boilerplate functions.
    """
    # functions to call before running the translation rule
    setup_funcs = (
        ensure_gpu_ext,
        setup_cuda_mpi,
        setup_hip_mpi,
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for f in setup_funcs:
            f()
        return func(*args, **kwargs)

    return wrapped
