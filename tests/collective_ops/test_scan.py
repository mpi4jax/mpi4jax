from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_scan():
    from mpi4jax import scan

    arr = jnp.ones((3, 2)) * rank

    res, _ = scan(arr, op=MPI.SUM)
    assert jnp.array_equal(res, jnp.ones((3, 2)) * sum(range(rank + 1)))


def test_scan_jit():
    from mpi4jax import scan

    arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: scan(x, op=MPI.SUM)[0])(arr)
    assert jnp.array_equal(res, jnp.ones((3, 2)) * sum(range(rank + 1)))


def test_scan_scalar():
    from mpi4jax import scan

    arr = rank
    res, _ = scan(arr, op=MPI.SUM)
    assert jnp.array_equal(res, sum(range(rank + 1)))


def test_scan_scalar_jit():
    from mpi4jax import scan

    arr = rank
    res = jax.jit(lambda x: scan(x, op=MPI.SUM)[0])(arr)
    assert jnp.array_equal(res, sum(range(rank + 1)))
