import jax
from jax import linear_util as lu
from jax.interpreters import xla


token_override_registry = {}  # Dict[Callable, Callable]
recursive_token_forwarding_registry = {}  # Dict[Primitive, Callable]


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, "length mismatch: {}".format(list(map(len, args)))
    return list(map(f, *args))


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


recursive_token_forwarding_registry[xla.xla_call_p] = xla_call_overrride


def scan_override(read, eqn, token):
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


recursive_token_forwarding_registry[jax.lax.scan_p] = scan_override


def while_override(read, eqn, token):
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


recursive_token_forwarding_registry[jax.lax.while_p] = while_override


def cond_override(read, eqn, token):
    _, bind_params = eqn.primitive.get_bind_params(eqn.params)
    branch_jaxprs = []
    cond_var = eqn.invars[0]
    other_vars = eqn.invars[1:]
    for branch in bind_params["branches"]:
        new_branch_fn = lambda token, *args: _token_forwarding(
            jax.core.jaxpr_as_fun(branch), token
        )(*args)
        new_jaxpr = jax.make_jaxpr(new_branch_fn)(token, *safe_map(read, other_vars))
        branch_jaxprs.append(new_jaxpr)
    bind_params["branches"] = tuple(branch_jaxprs)
    bind_params["linear"] = (False,) + bind_params["linear"]
    ans = eqn.primitive.bind(
        read(cond_var), token, *safe_map(read, other_vars), **bind_params
    )
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
            if eqn.primitive in recursive_token_forwarding_registry:
                rewrite = recursive_token_forwarding_registry[eqn.primitive]
                token, ans = rewrite(read, eqn, token)
            # Here, we are just reapplying the original operation if
            # not part of our communication protocol.
            # This code is mostly taken from jax.core.eval_jaxpr
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


def _token_forwarding(f, token):
    def wrapper(*args, **kwargs):
        jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
        return _override_tokens(jaxpr.jaxpr, jaxpr.consts, token, *args, **kwargs)

    return wrapper


def auto_tokenize(f, token=None):
    """Automatically manage tokens between all mpi4jax ops.

    Supports most JAX methods, including ones defined with `fori_loop`, `cond`, `jit`,
    `while_loop`, and `scan`. 

    .. note::

       This transforms all existing mpi4jax ops and overrides their token management
       completely. We do not recommend using this transform if you need any control
       over the token managment.

    Arguments:
        f: Any JAX function.
        token (Optional[Token]): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        A transformed version of `f` that automatically manages all mpi4jax tokens. 

    """
    def wrapper(*args, **kwargs):
        jaxpr, pytree = jax.make_jaxpr(f, return_shape=True)(*args, **kwargs)
        _, pytree = jax.tree_flatten(pytree)
        res = _override_tokens(jaxpr.jaxpr, jaxpr.consts, token, *args, **kwargs)
        return jax.tree_unflatten(pytree, res[1:])  # Drop the token.

    return wrapper
