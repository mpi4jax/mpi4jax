import jax
from jax._src.util import safe_map
from jax import linear_util as lu
from jax.interpreters import xla


token_override_registry = {}


def xla_call_overrride(read, eqn, token):
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    bind_params["donated_invars"] = (False,) + bind_params["donated_invars"]
    map_token = lambda func: lu.wrap_init(
        lambda token, *args: _token_forwarding(func.call_wrapped, token)(*args)
    )
    subfuns = safe_map(map_token, subfuns)
    ans = eqn.primitive.bind(
        *subfuns, token, *safe_map(read, eqn.invars), **bind_params
    )
    token = ans[0]
    ans = ans[1:]  # Drop the token.
    return token, ans


def scan_call_override(read, eqn, token):
    _, bind_params = eqn.primitive.get_bind_params(eqn.params)
    new_body_fn = lambda token, *args: _token_forwarding(
        jax.core.jaxpr_as_fun(bind_params["jaxpr"]), token
    )(*args)
    bind_params["jaxpr"] = jax.make_jaxpr(new_body_fn)(
        token, *safe_map(read, eqn.invars)
    )
    # Update bind_params to account for the additional token.
    bind_params["num_carry"] += 1
    bind_params["linear"] = (False,) + bind_params["linear"]
    ans = eqn.primitive.bind(token, *safe_map(read, eqn.invars), **bind_params)
    token = ans[0]
    ans = ans[1:]  # Drop the token.
    return token, ans


def while_call_override(read, eqn, token):
    _, bind_params = eqn.primitive.get_bind_params(eqn.params)
    new_body_fn = lambda token, *args: _token_forwarding(
        jax.core.jaxpr_as_fun(bind_params["body_jaxpr"]), token
    )(*args)
    bind_params["body_jaxpr"] = jax.make_jaxpr(new_body_fn)(
        token, *safe_map(read, eqn.invars)
    )
    # We use `auto_tokenize` here since the condition function
    # is forced to only return a boolean value and cannot return
    # a token.
    new_cond_fn = lambda token, *args: auto_tokenize(
        jax.core.jaxpr_as_fun(bind_params["cond_jaxpr"]), token
    )(*args)
    bind_params["cond_jaxpr"] = jax.make_jaxpr(new_cond_fn)(
        token, *safe_map(read, eqn.invars)
    )
    # Update bind_params to account for the additional token.
    ans = eqn.primitive.bind(token, *safe_map(read, eqn.invars), **bind_params)
    token = ans[0]
    ans = ans[1:]  # Drop the token.
    return token, ans


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
                token, ans = xla_call_overrride(read, eqn, token)
            elif eqn.primitive is jax.lax.scan_p:
                token, ans = scan_call_override(read, eqn, token)
            elif eqn.primitive is jax.lax.while_p:
                token, ans = while_call_override(read, eqn, token)
            else:
                subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
                ans = eqn.primitive.bind(
                    *subfuns, *safe_map(read, eqn.invars), **bind_params
                )
            if eqn.primitive.multiple_results:
                safe_map(write, eqn.outvars, ans)
            else:
                write(eqn.outvars[0], ans)
    return (token,) + tuple(safe_map(read, jaxpr.outvars))


def _token_forwarding(f, token=None):
    def wrapper(*args, **kwargs):
        jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
        return _override_tokens(jaxpr.jaxpr, jaxpr.consts, token, *args, **kwargs)

    return wrapper


def auto_tokenize(f, token=None):
    def wrapper(*args, **kwargs):
        jaxpr, pytree = jax.make_jaxpr(f, return_shape=True)(*args, **kwargs)
        _, pytree = jax.tree_flatten(pytree)
        res = _override_tokens(jaxpr.jaxpr, jaxpr.consts, token, *args, **kwargs)
        return jax.tree_unflatten(pytree, res[1:])  # Drop the token.

    return wrapper
