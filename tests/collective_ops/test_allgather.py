from mpi4py import MPI

import jax
import jax.numpy as jnp

import pytest

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_allgather():
    from mpi4jax import allgather

    arr = jnp.ones((3, 2)) * rank

    res, _ = allgather(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_allgather_jit():
    from mpi4jax import allgather

    arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: allgather(x)[0])(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_allgather_scalar():
    from mpi4jax import allgather

    arr = rank
    res, _ = allgather(arr)
    assert jnp.array_equal(res, jnp.arange(size))


def test_allgather_scalar_jit():
    from mpi4jax import allgather

    arr = jnp.array(rank)
    res = jax.jit(lambda x: allgather(x)[0])(arr)
    assert jnp.array_equal(res, jnp.arange(size))


@pytest.mark.skipif(jax.__version_info__ < (0, 4, 18), reason="requires jax 0.4.18")
def test_allgather_scalar_jit_extended():
    from mpi4jax import allgather

    arrs = jax.random.split(jax.random.key(12345), size)
    arr = arrs[rank]
    res = jax.jit(lambda x: allgather(x)[0])(arr)
    assert res.dtype == arr.dtype
    assert jnp.array_equal(res, arrs)
