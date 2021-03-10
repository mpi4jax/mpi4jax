#!/usr/bin/env python

"""
Run with

$ mpirun -n <nproc> python -m pytest .
"""

import pytest

# don't hog memory if running on GPU
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax  # noqa: E402
import jax.config  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.enable_omnistaging()

from mpi4py import MPI  # noqa: E402

print(MPI.get_vendor())

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

print("MPI rank = ", rank)
print("MPI size = ", size)


def test_allreduce_jit():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM)[0])(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_bcast_jit():
    from mpi4jax import Bcast

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    if rank != 0:
        _arr = _arr * 0

    res = jax.jit(lambda x: Bcast(x, root=0)[0])(arr)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_jit():
    from mpi4jax import Recv, Send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    @jax.jit
    def send_jit(x):
        Send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(lambda x: Recv(x, source=proc, tag=proc)[0])(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jit():
    from mpi4jax import Sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    res = jax.jit(lambda x, y: Sendrecv(x, y, source=other, dest=other)[0])(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
