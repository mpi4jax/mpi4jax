import threading
import functools
from contextlib import contextmanager

import jax
from jax.interpreters.xla import Token
from jax.core import Tracer

ctx = threading.local()
ctx.token_stack = []


@contextmanager
def token_context(token=None):
    if token is None:
        token = jax.lax.create_token()

    if not isinstance(token, (Token, Tracer)):
        raise TypeError("First argument to token_context must be a token or None")

    try:
        ctx.token_stack.append(token)
        yield

    finally:
        ctx.token_stack.pop()


def is_jitting():
    dummy = jax.lax.create_token()
    return isinstance(dummy, Tracer)


def inject_ctx_token(func):
    @functools.wraps(func)
    def wrapped(*args, token=None, **kwargs):
        if not ctx.token_stack:
            return func(*args, **kwargs, token=token)

        if token is None:
            token = ctx.token_stack[-1]

        if is_jitting() and not isinstance(token, Tracer):
            raise RuntimeError(
                "A token_context from non-JIT code cannot be used within JIT. "
                "Consider adding a token_context to your JITed function."
            )

        res = func(*args, **kwargs, token=token)

        if isinstance(res, tuple):
            new_token = res[-1]
        else:
            new_token = res

        ctx.token_stack[-1] = new_token

        return res

    return wrapped
