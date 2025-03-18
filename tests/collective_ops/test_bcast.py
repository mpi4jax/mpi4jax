from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_bcast():
    from mpi4jax import bcast

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    if rank != 0:
        _arr = _arr * 0

    res = bcast(_arr, root=0)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


def test_bcast_jit():
    from mpi4jax import bcast

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    if rank != 0:
        _arr = _arr * 0

    res = jax.jit(lambda x: bcast(x, root=0))(arr)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


def test_bcast_scalar():
    from mpi4jax import bcast

    arr = 1
    _arr = 1

    if rank != 0:
        _arr = _arr * 0

    res = bcast(_arr, root=0)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


def test_bcast_scalar_jit():
    from mpi4jax import bcast

    arr = 1
    _arr = 1

    if rank != 0:
        _arr = _arr * 0

    res = jax.jit(lambda x: bcast(x, root=0))(_arr)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)
