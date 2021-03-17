import os
import atexit
import warnings
import threading
import functools

# to avoid excessive overhead, we ensure that some functions run only once
_runtime_state = threading.local()
_runtime_state.platforms_to_flush = set()
_runtime_state.cuda_mpi_setup_done = False
_runtime_state.logging_setup_done = False


def ensure_platform_flush(platform):
    # at exit, we wait for all pending operations to finish
    # this prevents deadlocks (see mpi4jax#22)
    if platform in _runtime_state.platforms_to_flush:
        return

    _runtime_state.platforms_to_flush.add(platform)

    from .flush import flush

    flush_platform = functools.partial(flush, platform=platform)
    atexit.register(flush_platform)


def ensure_omnistaging():
    import jax.config

    if not jax.config.omnistaging_enabled:
        raise RuntimeError("mpi4jax requires JAX omnistaging to be enabled.")


def ensure_gpu_ext():
    from .xla_bridge import HAS_GPU_EXT

    if not HAS_GPU_EXT:
        raise ImportError(
            "The mpi4jax GPU extensions could not be imported. "
            "Please re-build mpi4jax with CUDA support and try again."
        )


def _is_truthy(str_val):
    return str_val.lower() in ("true", "1", "on")


def _is_falsy(str_val):
    return str_val.lower() in ("false", "0", "off")


def setup_bridge_logging():
    if _runtime_state.logging_setup_done:
        return

    _runtime_state.logging_setup_done = True

    from .xla_bridge import mpi_xla_bridge

    enable_logging = _is_truthy(os.getenv("MPI4JAX_DEBUG", ""))
    mpi_xla_bridge.set_logging(enable_logging)


def setup_cuda_mpi():
    if _runtime_state.cuda_mpi_setup_done:
        return

    _runtime_state.cuda_mpi_setup_done = True

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


def translation_rule_cpu(func):
    """XLA primitive translation rule on CPU for mpi4jax custom calls.

    This runs generic setup and boilerplate functions.
    """
    # functions to call before running the translation rule
    setup_funcs = (
        functools.partial(ensure_platform_flush, "cpu"),
        ensure_omnistaging,
        setup_bridge_logging,
    )

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
        functools.partial(ensure_platform_flush, "gpu"),
        ensure_omnistaging,
        setup_bridge_logging,
        setup_cuda_mpi,
    )

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for f in setup_funcs:
            f()
        return func(*args, **kwargs)

    return wrapped
