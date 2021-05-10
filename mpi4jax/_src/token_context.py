import threading
import functools
from contextlib import contextmanager

import jax
from jax.interpreters.xla import Token

ctx = threading.local()
ctx.token_stack = []


@contextmanager
def token_context(token=None):
    if token is None:
        token = jax.lax.create_token()

    try:
        ctx.token_stack.append(token)
        yield

    finally:
        ctx.token_stack.pop()


def is_jitting():
    dummy = jax.lax.create_token()
    return not isinstance(dummy, Token)


def inject_ctx_token(func):
    @functools.wraps(func)
    def wrapped(*args, token=None, **kwargs):
        if token is None and len(ctx.token_stack) > 0:
            token = ctx.token_stack[-1]

        if is_jitting() and isinstance(token, Token):
            raise RuntimeError(
                "A token_context from non-JIT code cannot be used within JIT. "
                "Consider moving token_context into your JITed function."
            )

        try:
            res = func(*args, **kwargs, token=token)

        except jax.core.UnexpectedTracerError as exc:
            if not ctx.token_stack:
                # token_context is not in use
                raise

            raise RuntimeError(
                "Encountered unexpected tracer. Make sure not to use "
                "token_context across more than one JIT function."
            ) from exc

        else:
            if isinstance(res, tuple):
                new_token = res[-1]
            else:
                new_token = res

            ctx.token_stack[-1] = new_token

        return res

    return wrapped
