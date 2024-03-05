import os
import warnings
import functools

# global variables to keep track of state
_cuda_mpi_setup_done = False
_sycl_mpi_setup_done = False


def ensure_gpu_ext():
    from .xla_bridge import HAS_GPU_EXT

    if not HAS_GPU_EXT:
        raise ImportError(
            "The mpi4jax GPU extensions could not be imported. "
            "Please re-build mpi4jax with CUDA support and try again."
        )


def ensure_xpu_ext():
    from .xla_bridge import HAS_XPU_EXT

    if not HAS_XPU_EXT:
        raise ImportError(
            "The mpi4jax XPU extensions could not be imported. "
            "Please re-build mpi4jax with SYCL support and try again."
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

    from .xla_bridge import mpi_xla_bridge_gpu

    mpi_xla_bridge_gpu.set_copy_to_host(not has_cuda_mpi)


def setup_sycl_mpi():
    global _sycl_mpi_setup_done

    if _sycl_mpi_setup_done:
        return

    _sycl_mpi_setup_done = True

    xpu_copy_behavior = os.getenv("MPI4JAX_USE_SYCL_MPI", "")

    if _is_truthy(xpu_copy_behavior):
        has_sycl_mpi = True
    elif _is_falsy(xpu_copy_behavior):
        has_sycl_mpi = False
    else:
        has_sycl_mpi = False
        warn_msg = (
            "Not using SYCL-enabled MPI. "
            "If you are sure that your MPI library is built with SYCL support, "
            "set MPI4JAX_USE_SYCL_MPI=1. To silence this warning, "
            "set MPI4JAX_USE_SYCL_MPI=0."
        )
        warnings.warn(warn_msg)

    from .xla_bridge import mpi_xla_bridge_xpu

    mpi_xla_bridge_xpu.set_copy_to_host(not has_sycl_mpi)


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
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for f in setup_funcs:
            f()
        return func(*args, **kwargs)

    return wrapped


def translation_rule_xpu(func):
    """XLA primitive translation rule on XPU for mpi4jax custom calls.

    This runs generic setup and boilerplate functions.
    """
    # functions to call before running the translation rule
    setup_funcs = (
        ensure_xpu_ext,
        setup_sycl_mpi,
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for f in setup_funcs:
            f()
        return func(*args, **kwargs)

    return wrapped
