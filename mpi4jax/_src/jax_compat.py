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


jax_version = versiontuple(jax.__version__)

if jax_version >= versiontuple("0.2.26"):
    from jax.core import Tracer, Token
else:
    from jax.core import Tracer
    from jax.interpreters.xla import Token


if jax_version >= versiontuple("0.3.15"):
    # abstract eval needs to return effects
    # see https://github.com/google/jax/issues/11620
    from functools import wraps
    from jax.interpreters import mlir
    from jax._src.lax import control_flow as lcf

    class MPIEffect:
        def __hash__(self):
            # enforce a constant (known) hash
            return hash("I love mpi4jax")

    effect = MPIEffect()
    mlir.lowerable_effects.add(effect)
    lcf.allowed_effects.add(effect)

    def register_abstract_eval(primitive, func):
        """Injects an effect object into func and registers it via `primitive.def_effectful_abstract_eval`.
        
        (as required by JAX>=0.3.15 to ensure primitives are staged out)
        """
        @wraps(func)
        def effects_wrapper(*args, **kwargs):
            return func(*args, **kwargs), {effect}

        primitive.def_effectful_abstract_eval(effects_wrapper)

else:
    # TODO: drop this path when we require jax>=0.3.15

    def register_abstract_eval(primitive, func):
        primitive.def_abstract_eval(func)


__all__ = [
    "Tracer",
    "Token",
]


def check_jax_version():
    # check version of jaxlib
    import jaxlib

    if versiontuple(jaxlib.__version__) < versiontuple(JAXLIB_MINIMUM_VERSION):
        raise RuntimeError(
            f"mpi4jax requires jaxlib>={JAXLIB_MINIMUM_VERSION}, but you have "
            f"{jaxlib.__version__}. Please install a supported version of JAX and jaxlib."
        )
