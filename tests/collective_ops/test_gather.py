from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_gather():
    from mpi4jax import gather

    arr = jnp.ones((3, 2)) * rank

    res, _ = gather(arr, root=0)
    if rank == 0:
        for p in range(size):
            assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)
    else:
        assert jnp.array_equal(res, arr)


def test_gather_jit():
    from mpi4jax import gather

    arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: gather(x, root=0)[0])(arr)
    if rank == 0:
        for p in range(size):
            assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)
    else:
        assert jnp.array_equal(res, arr)


def test_gather_scalar():
    from mpi4jax import gather

    arr = rank
    res, _ = gather(arr, root=0)
    if rank == 0:
        assert jnp.array_equal(res, jnp.arange(size))
    else:
        assert jnp.array_equal(res, arr)


def test_gather_scalar_jit():
    from mpi4jax import gather

    arr = rank
    res = jax.jit(lambda x: gather(x, root=0)[0])(arr)
    if rank == 0:
        assert jnp.array_equal(res, jnp.arange(size))
    else:
        assert jnp.array_equal(res, arr)
