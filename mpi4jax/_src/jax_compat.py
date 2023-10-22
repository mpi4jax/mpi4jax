import os
import re
import warnings

import jax
import jaxlib

from jax.interpreters.mlir import token_type  # noqa: F401


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


# TODO: remove the other path once we require jax/lib > 0.4.16
if versiontuple(jax.__version__) >= (0, 4, 16):
    from jax.interpreters.mlir import custom_call  # noqa: F401
else:
    from jaxlib.hlo_helpers import custom_call as _custom_call

    # Recent versions return a structure with a field 'results'. We mock it on
    # older versions
    from collections import namedtuple

    MockResult = namedtuple("MockResult", ["results"])

    def custom_call(*args, result_types, **kwargs):
        results = _custom_call(*args, out_types=result_types, **kwargs)
        # TODO: remove this path once we require jax>=0.4.10
        if versiontuple(jaxlib.__version__) < (0, 4, 10):
            if not isinstance(results, list):
                results = [results]
        return MockResult(results)


# TODO: remove this code once we only support jax > 0.4.14
if versiontuple(jax.__version__) >= (0, 4, 14):
    from jax.core import ShapedArray  # noqa: F401
else:
    from jax.abstract_arrays import ShapedArray  # noqa: F401


# TODO: remove this code once we only support jax >= 0.4.16
if versiontuple(jax.__version__) >= (0, 4, 16):
    EffectType = jax._src.effects.Effect

    def register_effect(EffectType, ordered=False):
        from jax._src.effects import (
            lowerable_effects,
            ordered_effects,
            control_flow_allowed_effects,
            custom_derivatives_allowed_effects,
        )

        effect = EffectType()
        lowerable_effects.add_type(EffectType)

        if ordered:
            ordered_effects.add_type(EffectType)

        control_flow_allowed_effects.add_type(EffectType)
        # Effects must be added to the allow_effects list in order to work within
        # custom_vjp. See google/jax#11916
        custom_derivatives_allowed_effects.add_type(EffectType)
        return effect

else:
    EffectType = object

    def register_effect(EffectType, ordered=False):
        from jax.interpreters import mlir
        from jax._src.lax import control_flow as lcf
        import jax._src.custom_derivatives as custom_derivatives

        if ordered:
            # orderd effects are not supported, ensure that it is not used
            return None

        effect = EffectType()
        mlir.lowerable_effects.add_type(EffectType)
        lcf.allowed_effects.add_type(EffectType)
        # Effects must be added to the allow_effects list in order to work within
        # custom_vjp. See google/jax#11916
        custom_derivatives.allowed_effects.add_type(EffectType)
        return effect
