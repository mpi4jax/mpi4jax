import atexit
import threading
import functools

from .flush import flush


_runtime_state = threading.local()
_runtime_state.platforms_to_flush = set()


def ensure_platform_flush(platform):
    # at exit, we wait for all pending operations to finish
    # this prevents deadlocks (see mpi4jax#22)
    if platform in _runtime_state.platforms_to_flush:
        return

    _runtime_state.platforms_to_flush.add(platform)
    flush_platform = functools.partial(flush, platform=platform)
    atexit.register(flush_platform)


def ensure_omnistaging():
    import jax.config

    if not jax.config.omnistaging_enabled:
        raise RuntimeError("mpi4jax requires JAX omnistaging to be enabled.")


def translation_rule_cpu(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        ensure_platform_flush("cpu")
        ensure_omnistaging()
        return func(*args, **kwargs)

    return wrapped


def translation_rule_gpu(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        ensure_platform_flush("gpu")
        ensure_omnistaging()
        return func(*args, **kwargs)

    return wrapped
