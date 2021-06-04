import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    res, token = sendrecv(arr, arr, source=other, dest=other)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_status():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    status = MPI.Status()
    res, token = sendrecv(arr, arr, source=other, dest=other, status=status)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
    assert status.Get_source() == other


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_status_jit():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    status = MPI.Status()
    res = jax.jit(
        lambda x, y: sendrecv(x, y, source=other, dest=other, status=status)[0]
    )(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
    assert status.Get_source() == other


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_scalar():
    from mpi4jax import sendrecv

    arr = 1 * rank
    _arr = arr

    other = 1 - rank

    res, token = sendrecv(arr, arr, source=other, dest=other)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jit():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    res = jax.jit(lambda x, y: sendrecv(x, y, source=other, dest=other)[0])(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_scalar_jit():
    from mpi4jax import sendrecv

    arr = 1 * rank
    _arr = arr

    other = 1 - rank

    res = jax.jit(lambda x, y: sendrecv(x, y, source=other, dest=other)[0])(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_vmap():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    res = sendrecv(arr, arr, source=other, dest=other)[0]

    def fun(x, y):
        return sendrecv(x, y, source=other, dest=other)[0]

    vfun = jax.vmap(fun, in_axes=(0, 0))
    res = vfun(_arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_grad():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * (rank + 1)
    _arr = arr.copy()

    other = 1 - rank

    def f(x):
        x, token = sendrecv(x, x, source=other, dest=other)
        x = x * (rank + 1)
        return x.sum()

    res = jax.grad(f)(arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * (other + 1))
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_grad_2():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * (rank + 1)
    _arr = arr.copy()

    other = 1 - rank

    def f(x):
        x, token = sendrecv(x, x, source=other, dest=other)
        x = x * (rank + 1) * 5
        x, token = sendrecv(x, x, source=other, dest=other, token=token)
        x = x * (rank + 1) ** 2
        return x.sum()

    res = jax.grad(f)(arr)

    solution = (rank + 1) ** 2 * (other + 1) * 5
    print("solution is ", solution)
    assert jnp.array_equal(res, jnp.ones_like(arr) * solution)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jacfwd():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * (rank + 1)
    # _arr = arr.copy()

    other = 1 - rank

    def f(x):
        x, token = sendrecv(x, x, source=other, dest=other)
        x = x * (rank + 1)
        return x.sum()

    with pytest.raises(RuntimeError):
        jax.jacfwd(f)(arr)

    # assert jnp.array_equal(res, jnp.ones_like(arr) * (other + 1))
    # assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jacrev():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * (rank + 1)
    _arr = arr.copy()

    other = 1 - rank

    def f(x):
        x, token = sendrecv(x, x, source=other, dest=other)
        x = x * (rank + 1)
        return x.sum()

    res = jax.jacrev(f)(arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * (other + 1))
    assert jnp.array_equal(_arr, arr)
