import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            res = recv(arr, source=proc, tag=proc)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_scalar():
    from mpi4jax import recv, send

    arr = 1 * rank
    _arr = 1 * rank

    if rank == 0:
        for proc in range(1, size):
            res = recv(arr, source=proc, tag=proc)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_scalar_jit():
    from mpi4jax import recv, send

    arr = 1 * rank
    _arr = 1 * rank

    @jax.jit
    def send_jit(x):
        send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(lambda x: recv(x, source=proc, tag=proc))(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_jit():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    @jax.jit
    def send_jit(x):
        send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(lambda x: recv(x, source=proc, tag=proc))(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_send_recv_deadlock():
    from mpi4jax import recv, send

    # this deadlocks without proper token management
    @jax.jit
    def deadlock(arr):
        if rank == 0:
            # send, then receive
            send(arr, 1)
            newarr = recv(arr, 1)
        else:
            # receive, then send
            newarr = recv(arr, 0)
            send(arr, 0)
        return newarr

    arr = jnp.ones(10) * rank
    arr = deadlock(arr)
    assert jnp.array_equal(arr, jnp.ones_like(arr) * (1 - rank))


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_status():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res = recv(arr, source=proc, tag=proc, status=status)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_status_jit():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    @jax.jit
    def send_jit(x):
        send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res = jax.jit(lambda x: recv(x, source=proc, tag=proc, status=status))(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)
