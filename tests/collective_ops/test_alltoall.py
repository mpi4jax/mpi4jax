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


def test_alltoall_transpose():
    # test whether consecutive transposes are behaving correctly under JIT
    # (some backends can optimize them out which leads to non-c-contiguous arrays)
    # see mpi4jax#176
    from mpi4jax import alltoall

    shape = (256, 256)

    def transpose(arr):
        arr = arr.reshape([shape[0] // size, size, shape[1] // size])
        arr = arr.transpose([1, 0, 2])
        arr, token = alltoall(arr, comm=comm)
        arr = arr.reshape([shape[0], shape[1] // size])
        arr = arr.transpose([1, 0])
        return arr, token

    transpose_jit = jax.jit(transpose)

    master_key = jax.random.PRNGKey(42)
    key = jax.random.split(master_key, size)[rank]
    arr = jax.random.normal(key, [shape[0] // size, shape[1]])

    assert jnp.array_equal(transpose(arr)[0], transpose_jit(arr)[0])
