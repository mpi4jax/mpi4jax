import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_alltoall():
    from mpi4jax import Alltoall

    arr = jnp.ones((size, 3, 2)) * rank

    res, _ = Alltoall(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_alltoall_jit():
    from mpi4jax import Alltoall

    arr = jnp.ones((size, 3, 2)) * rank

    res = jax.jit(lambda x: Alltoall(x)[0])(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_alltoall_wrong_size():
    from mpi4jax import Alltoall

    arr = jnp.ones((size + 1,)) * rank
    with pytest.raises(ValueError) as excinfo:
        res, _ = Alltoall(arr)

    assert "must be divisible by number of processes" in str(excinfo.value)
