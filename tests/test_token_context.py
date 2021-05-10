import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp


def test_token_context():
    from mpi4jax import allreduce, token_context
    from mpi4jax._src.token_context import ctx

    arr = jnp.ones((3, 2))

    # with initial token
    initial_token = jax.lax.create_token()
    expected_addr = id(initial_token)

    assert ctx.token_stack == []

    with token_context(initial_token):
        assert ctx.token_stack[0] is initial_token

        res, token = allreduce(arr, op=MPI.SUM)
        assert id(token) == expected_addr
        res, token2 = allreduce(arr * 2, op=MPI.SUM)
        assert id(token2) == expected_addr

    assert ctx.token_stack == []

    # without initial token
    with token_context():
        expected_addr = id(ctx.token_stack[-1])

        res, token = allreduce(arr, op=MPI.SUM)
        assert id(token) == expected_addr
        res, token2 = allreduce(arr * 2, op=MPI.SUM)
        assert id(token2) == expected_addr

    assert ctx.token_stack == []


def test_token_context_nested():
    from mpi4jax import allreduce, token_context
    from mpi4jax._src.token_context import ctx

    arr = jnp.ones((3, 2))
    initial_token = jax.lax.create_token()

    with token_context(initial_token):
        assert ctx.token_stack[0] is initial_token

        res, token = allreduce(arr, op=MPI.SUM)
        res, token2 = allreduce(arr * 2, op=MPI.SUM)

        with token_context(token2):
            assert len(ctx.token_stack) == 2
            assert ctx.token_stack[1] is token2

            res, token3 = allreduce(arr * 3, op=MPI.SUM)
            assert ctx.token_stack[0] is token2
            assert ctx.token_stack[1] is token3

    assert ctx.token_stack == []


def test_token_context_jit():
    from mpi4jax import allreduce, token_context
    from mpi4jax._src.token_context import ctx

    arr = jnp.ones((3, 2))

    @jax.jit
    def bar(arr, initial_token=None):
        with token_context(initial_token):
            res1, token1 = allreduce(arr, op=MPI.SUM)
            assert ctx.token_stack[-1] is token1
            res2, token2 = allreduce(arr * 2, op=MPI.SUM)
            assert ctx.token_stack[-1] is token2

        return res1, res2

    bar(arr)
    assert ctx.token_stack == []

    with token_context():
        res, token = allreduce(arr, op=MPI.SUM)
        bar(arr, token)

    assert ctx.token_stack == []


def test_token_context_decorator():
    from mpi4jax import allreduce, token_context
    from mpi4jax._src.token_context import ctx

    arr = jnp.ones((3, 2))

    @jax.jit
    @token_context()
    def bar(arr):
        res1, token1 = allreduce(arr, op=MPI.SUM)
        assert ctx.token_stack[-1] is token1
        res2, token2 = allreduce(arr * 2, op=MPI.SUM)
        assert ctx.token_stack[-1] is token2
        return res1, res2

    bar(arr)
    assert ctx.token_stack == []


def test_token_context_leak():
    from mpi4jax import allreduce, token_context

    arr = jnp.ones((3, 2))

    @jax.jit
    def foo(arr):
        res1, _ = allreduce(arr, op=MPI.SUM)
        res2, _ = allreduce(arr * 2, op=MPI.SUM)
        return res1, res2

    with pytest.raises(RuntimeError) as excinfo:
        with token_context():
            foo(arr)

    assert "from non-JIT code cannot be used within JIT" in str(excinfo.value)

    # decorator: wrong order (jit must be first)

    @token_context()
    @jax.jit
    def bar(arr):
        res1, _ = allreduce(arr, op=MPI.SUM)
        return res1

    with pytest.raises(RuntimeError) as excinfo:
        bar(arr)

    assert "from non-JIT code cannot be used within JIT" in str(excinfo.value)
