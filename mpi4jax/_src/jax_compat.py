import os
import re
import warnings

import jax
import jaxlib


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


# TODO: remove this code once we only support jax/lib > 0.4.2
if versiontuple(jaxlib.__version__) >= (0, 4, 2):
    from jaxlib.hlo_helpers import custom_call as hlo_custom_call  # noqa: F401
else:
    from jaxlib.mhlo_helpers import custom_call as hlo_custom_call  # noqa: F401


# TODO: remove this code once we only support jax > 0.4.4
if versiontuple(jax.__version__) >= (0, 4, 4):
    from jax._src.interpreters.mlir import token_type  # noqa: F401
else:
    from jax.interpreters.mlir import token_type  # noqa: F401


# TODO: remove this code once we only support jax > 0.4.5
if versiontuple(jax.__version__) >= (0, 4, 5):

    def register_effect(EffectType):
        from jax.interpreters import mlir
        from jax._src.lax import control_flow as lcf
        import jax._src.custom_derivatives as custom_derivatives

        effect = EffectType()
        mlir.lowerable_effects.add_type(EffectType)
        lcf.allowed_effects.add_type(EffectType)
        # Effects must be added to the allow_effects list in order to work within
        # custom_vjp. See google/jax#11916
        custom_derivatives.allowed_effects.add_type(EffectType)
        return effect

else:

    def register_effect(EffectType):
        from jax.interpreters import mlir
        from jax._src.lax import control_flow as lcf
        import jax._src.custom_derivatives as custom_derivatives

        effect = EffectType()
        mlir.lowerable_effects.add(effect)
        lcf.allowed_effects.add(effect)
        # Effects must be added to the allow_effects list in order to work within
        # custom_vjp. See google/jax#11916
        custom_derivatives.allowed_effects.add(effect)
        return effect
