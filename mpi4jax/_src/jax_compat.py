import jax

JAXLIB_MINIMUM_VERSION = "0.1.62"


def versiontuple(verstr):
    # drop everything after the numeric part of the version
    allowed_chars = "0123456789."
    for i, char in enumerate(verstr):
        if char not in allowed_chars:
            break
    else:
        i = len(verstr) + 1

    verstr = verstr[:i].rstrip(".")
    return tuple(int(v) for v in verstr.split("."))[:3]


def check_jax_version():
    # check version of jaxlib
    import jaxlib

    if versiontuple(jaxlib.__version__) < versiontuple(JAXLIB_MINIMUM_VERSION):
        raise RuntimeError(
            f"mpi4jax requires jaxlib>={JAXLIB_MINIMUM_VERSION}, but you have "
            f"{jaxlib.__version__}. Please install a supported version of JAX and jaxlib."
        )


jax_version = versiontuple(jax.__version__)

if jax_version >= versiontuple("0.2.26"):
    from jax.core import Tracer, Token
else:
    from jax.core import Tracer
    from jax.interpreters.xla import Token


__all__ = [
    "Tracer",
    "Token",
]
