import os
import re
import warnings

import jax

from jax.interpreters import mlir
from jax.interpreters.mlir import TokenSet
from jax.extend.core import Primitive  # noqa: F401


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


def register_custom_call_target(name, fn, *, platform: str, api_version: int = 1):
    """Register a custom call target with XLA.

    Args:
        name: Name of the custom call target.
        fn: PyCapsule containing the function pointer.
        platform: Platform to register for (e.g., "cpu", "CUDA", "SYCL").
        api_version: API version (0 for legacy, 1 for new FFI API). Defaults to 1.
    """
    return jax.ffi.register_ffi_target(
        name, fn, platform=platform, api_version=api_version
    )


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


if versiontuple(jax.__version__) >= (0, 11, 0):
    # jax >= 0.11.0 re-exposes the public singleton here (jax.core.abstract_token
    # was deprecated in 0.10.0 and removed in 0.11.0).
    from jax.ffi import abstract_token  # noqa: F401
elif versiontuple(jax.__version__) >= (0, 10, 0):
    # jax 0.10.x: the public jax.core.abstract_token is deprecated. Use the
    # private singleton to avoid a DeprecationWarning. We must reuse the
    # singleton rather than constructing a fresh AbstractToken(): JAX identifies
    # tokens via `aval is core.abstract_token` in its lowering code, so a new
    # instance would fail those checks.
    from jax._src.core import abstract_token  # noqa: F401
else:
    from jax.core import abstract_token  # noqa: F401
