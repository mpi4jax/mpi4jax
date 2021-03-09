from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_allgather():
    from mpi4jax import Allgather

    arr = jnp.ones((3, 2)) * rank

    res, _ = Allgather(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_allgather_jit():
    from mpi4jax import Allgather

    arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: Allgather(x)[0])(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_allgather_scalar():
    from mpi4jax import Allgather

    arr = rank
    res, _ = Allgather(arr)
    assert jnp.array_equal(res, jnp.arange(size))


def test_allgather_scalar_jit():
    from mpi4jax import Allgather

    arr = rank
    res = jax.jit(lambda x: Allgather(x)[0])(arr)
    assert jnp.array_equal(res, jnp.arange(size))
