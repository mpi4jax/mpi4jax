import jax
from jax._src.util import safe_map

token_override_registry = {}

def _override_tokens(jaxpr, consts, token, *args):
    if token is None:
        token = jax.core.create_token()
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
        if eqn.primitive in token_override_registry:
            token_override = token_override_registry[eqn.primitive]
            ans = token_override(
                safe_map(read, eqn.invars), new_token=token, **eqn.params)
            token = ans[-1]
            safe_map(write, eqn.outvars, ans)
        else:
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            ans = eqn.primitive.bind(
                *subfuns, *safe_map(read, eqn.invars), **bind_params)
            if eqn.primitive.multiple_results:
                safe_map(write, eqn.outvars, ans)
            else:
                write(eqn.outvars[0], ans)
    res = safe_map(read, jaxpr.outvars)
    if len(res) == 1:
        return res[0]
    return res

def auto_tokenize(f, token=None):
    if token is None:
        token = jax.lax.create_token()
    def wrapper(*args, **kwargs):
        jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
        return _override_tokens(jaxpr.jaxpr, jaxpr.consts, token, *args, **kwargs)

    return wrapper