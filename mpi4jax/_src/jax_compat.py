import os
import re
import warnings

import jax


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


def check_jax_version():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "_latest_jax_version.txt")) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            pkg, jax_maxversion = re.match(r"(\w+)\b==(.*)", line).groups()
            assert pkg == "jax"
            break

    warn_envvar = "MPI4JAX_NO_WARN_JAX_VERSION"
    nowarn = os.environ.get(warn_envvar, "").lower() in ("1", "true", "on")

    if not nowarn and versiontuple(jax.__version__) > versiontuple(jax_maxversion):
        warnings.warn(
            f"\nThe latest supported JAX version with this release of mpi4jax is {jax_maxversion}, "
            f"but you have {jax.__version__}. If you encounter problems consider downgrading JAX, "
            f"for example via:\n\n"
            f"    $ pip install jax[cpu]=={jax_maxversion}\n\n"
            f"Or try upgrading mpi4jax via\n\n"
            f"    $ pip install -U mpi4jax\n\n"
            f"You can set the environment variable `{warn_envvar}=1` to silence this warning."
        )
