from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_reduce():
    from mpi4jax import reduce

    arr = jnp.ones((3, 2)) * rank

    res, _ = reduce(arr, op=MPI.SUM, root=0)
    if rank == 0:
        assert jnp.array_equal(res, jnp.ones((3, 2)) * sum(range(size)))
    else:
        assert jnp.array_equal(res, arr)


def test_reduce_jit():
    from mpi4jax import reduce

    arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: reduce(x, op=MPI.SUM, root=0)[0])(arr)
    if rank == 0:
        assert jnp.array_equal(res, jnp.ones((3, 2)) * sum(range(size)))
    else:
        assert jnp.array_equal(res, arr)


def test_reduce_scalar():
    from mpi4jax import reduce

    arr = rank
    res, _ = reduce(arr, op=MPI.SUM, root=0)
    if rank == 0:
        assert jnp.array_equal(res, sum(range(size)))
    else:
        assert jnp.array_equal(res, arr)


def test_reduce_scalar_jit():
    from mpi4jax import reduce

    arr = rank
    res = jax.jit(lambda x: reduce(x, op=MPI.SUM, root=0)[0])(arr)
    if rank == 0:
        assert jnp.array_equal(res, sum(range(size)))
    else:
        assert jnp.array_equal(res, arr)
