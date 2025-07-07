import os
import re
import warnings

import jax

from jax.interpreters import mlir
from jax.interpreters.mlir import token_type as jax_token_type, TokenSet
from jax.extend.core import Primitive, Token  # noqa: F401


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


def register_lowering(prim, rule, platform="cpu"):
    try:
        return mlir.register_lowering(prim, rule, platform=platform)
    except NotImplementedError:
        # Raised if the platform is supplied by a non-installed plugin
        assert platform != "cpu"
        return None


def register_custom_call_target(name, fn, *, platform: str, api_version: int):
    return jax.ffi.register_ffi_target(
        name, fn, platform=platform, api_version=api_version
    )


token_type = jax_token_type
get_token_effect = lambda ctx, effect: ctx.tokens_in.get(effect)
set_token_effect = lambda ctx, effect, token: ctx.set_tokens_out(
    TokenSet({effect: token})
)

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
