import jax
from jax import linear_util as lu
from jax.interpreters import xla

# registry for wrapped mpi4jax ops
from .register_overrides import token_override_registry  # noqa: E402

# registry for wrapped JAX primitives
recursive_token_forwarding_registry = {}


def safe_map(f, *args):
    args = [list(arg) for arg in args]
    for arg in args:
        assert len(arg) == len(args[0]), f"Mismatched lengths: {[len(x) for x in args]}"
    return [f(*arg) for arg in zip(*args)]


def xla_call_overrride(read, eqn, token):
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    bind_params["donated_invars"] = (False,) + bind_params["donated_invars"]
    map_token = lambda func: lu.wrap_init(  # noqa: E731
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
    new_body_fn = lambda token, *args: _token_forwarding(  # noqa: E731
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
    new_body_fn = lambda token, *args: _token_forwarding(  # noqa: E731
        jax.core.jaxpr_as_fun(bind_params["body_jaxpr"]), token
    )(*args)
    bind_params["body_jaxpr"] = jax.make_jaxpr(new_body_fn)(
        token, *safe_map(read, eqn.invars)
    )
    # We use `auto_tokenize` here since the condition function
    # is forced to only return a boolean value and cannot return
    # a token.
    new_cond_fn = lambda token, *args: _token_forwarding(  # noqa: E731
        jax.core.jaxpr_as_fun(bind_params["cond_jaxpr"]), token
    )(*args)[1:]
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
        new_branch_fn = lambda token, *args: _token_forwarding(  # noqa: E731
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


recursive_token_forwarding_registry[jax.lax.cond_p] = cond_override


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

    # compatibility with jax<0.3.10
    if hasattr(jax.core, "unitvar"):
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


def auto_tokenize(f):
    """Automatically manage tokens between all mpi4jax ops.

    Supports most JAX operations, including ``fori_loop``, ``cond``, ``jit``,
    ``while_loop``, and ``scan``. This transforms *all* operations in the decorated
    function, even through subfunctions and nested applications of ``jax.jit``.

    .. note::

       This transforms overrides all mpi4jax token management completely.
       Do not use this transform if you need manual control over the token managment.

    Arguments:
        f: Any function that uses mpi4jax primitives (jitted or not).

    Returns:
        A transformed version of ``f`` that automatically manages all mpi4jax tokens.

    Example:

        >>> @auto_tokenize
        ... def f(a):
        ...     # no token handling necessary
        ...     res, _ = allreduce(a, op=MPI.SUM)
        ...     res, _ = allreduce(res, op=MPI.SUM)
        ...     return res
        >>> arr = jnp.ones((3, 2))
        >>> res = f(arr)

    """

    def wrapper(*args, **kwargs):
        jaxpr, pytree = jax.make_jaxpr(f, return_shape=True)(*args, **kwargs)
        _, pytree = jax.tree_util.tree_flatten(pytree)
        res = _override_tokens(jaxpr.jaxpr, jaxpr.consts, None, *args, **kwargs)
        return jax.tree_util.tree_unflatten(pytree, res[1:])  # Drop the token.

    return wrapper
