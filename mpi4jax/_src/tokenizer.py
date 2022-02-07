import jax
from jax._src.util import safe_map
from jax import linear_util as lu
from jax.interpreters import xla


token_override_registry = {}


def _override_tokens(jaxpr, consts, token, *args):
    if token is None:  # Create a new token if one is not passed.
        token = jax.lax.create_token()

    def read(v):
        if type(v) is jax.core.Literal:
            return v.val
        else:
            return env[v]

    def write(v, val):
        env[v] = val

    env = {}
    write(jax.core.unitvar, jax.core.unit)
    safe_map(write, jaxpr.constvars, consts)
    safe_map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        # Here, we override the original primatives with our own
        # token forcing method.
        if eqn.primitive in token_override_registry:
            token_override = token_override_registry[eqn.primitive]
            ans = token_override(
                safe_map(read, eqn.invars), new_token=token, **eqn.params
            )
            # We pass along the new token returned by the above binding
            token = ans[-1]
            safe_map(write, eqn.outvars, ans)
        else:
            # Here, we are just reapplying the original operation if
            # not part of our communication protocol.
            # This code is mostly taken form jax.core.eval_jaxpr
            if eqn.primitive is xla.xla_call_p:
                subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
                bind_params["donated_invars"] = (False,) + bind_params["donated_invars"]
                map_token = lambda x: lu.wrap_init(
                    lambda token, *args: _auto_tokenize(x.call_wrapped, token)(*args)
                )
                subfuns = safe_map(map_token, subfuns)
                ans = eqn.primitive.bind(
                    *subfuns, token, *safe_map(read, eqn.invars), **bind_params
                )
                token = ans[-1]
                ans = ans[:-1]  # Drop the token.
            else:
                subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
                # subfuns = safe_map(lambda x: lu.wrap_init(_auto_tokenize(x.call_wrapped)), subfuns)

                ans = eqn.primitive.bind(
                    *subfuns, *safe_map(read, eqn.invars), **bind_params
                )
            if eqn.primitive.multiple_results:
                safe_map(write, eqn.outvars, ans)
            else:
                write(eqn.outvars[0], ans)
    return tuple(safe_map(read, jaxpr.outvars)) + (token,)


def _auto_tokenize(f, token=None):
    def wrapper(*args, **kwargs):
        jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
        return _override_tokens(jaxpr.jaxpr, jaxpr.consts, token, *args, **kwargs)

    return wrapper


def auto_tokenize(f, token=None):
    def wrapper(*args, **kwargs):
        res = _auto_tokenize(f)(*args, **kwargs)
        res = res[:-1]  # Drop the token.
        if len(res) == 1:
            return res[0]
        return tuple(res)

    return wrapper
