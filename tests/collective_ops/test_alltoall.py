import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_alltoall():
    from mpi4jax import alltoall

    arr = jnp.ones((size, 3, 2)) * rank

    res, _ = alltoall(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_alltoall_jit():
    from mpi4jax import alltoall

    arr = jnp.ones((size, 3, 2)) * rank

    res = jax.jit(lambda x: alltoall(x)[0])(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_alltoall_wrong_size():
    from mpi4jax import alltoall

    arr = jnp.ones((size + 1,)) * rank
    with pytest.raises(ValueError) as excinfo:
        alltoall(arr)

    assert "must have shape (nproc, ...)" in str(excinfo.value)
