#!/usr/bin/env python

"""
Run with

$ mpirun -n <nproc> python -m pytest .
"""

import pytest

import jax
import jax.config
import jax.numpy as np
jax.config.enable_omnistaging()

from mpi4py import MPI  # noqa: E402

print(MPI.get_vendor())

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

print("MPI rank = ", rank)
print("MPI size = ", size)


def test_allreduce():
    from mpi4jax import Allreduce

    arr = np.ones((3, 2))
    _arr = arr.copy()

    res = Allreduce(arr, op=MPI.SUM)
    assert np.array_equal(res, arr * size)
    assert np.array_equal(_arr, arr)

    res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM))(arr)
    assert np.array_equal(res, arr * size)
    assert np.array_equal(_arr, arr)

    res = jax.vmap(lambda x: Allreduce(x, op=MPI.SUM), in_axes=0)(arr)
    assert np.array_equal(res, arr * size)
    assert np.array_equal(_arr, arr)

    res = jax.jit(jax.vmap(lambda x: Allreduce(x, op=MPI.SUM), in_axes=0))(arr)
    assert np.array_equal(res, arr * size)
    assert np.array_equal(_arr, arr)

    res, grad = jax.value_and_grad(lambda x: Allreduce(x, op=MPI.SUM).sum())(arr)
    assert np.array_equal(res, arr.sum() * size)
    assert np.array_equal(grad, np.ones(arr.shape))
    assert np.array_equal(_arr, arr)

    res, grad = jax.jit(jax.value_and_grad(lambda x: Allreduce(x, op=MPI.SUM).sum()))(
        arr
    )
    assert np.array_equal(res, arr.sum() * size)
    assert np.array_equal(grad, np.ones(arr.shape))
    assert np.array_equal(_arr, arr)

    with pytest.raises(NotImplementedError):
        jax.jit(jax.value_and_grad(lambda x: Allreduce(x, op=MPI.MIN).sum()))(arr)


def test_send_recv():
    from mpi4jax import Send, Recv

    arr = np.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            res = Recv(arr, source=proc, tag=proc)
            assert np.array_equal(res, np.ones_like(arr) * proc)
            assert np.array_equal(_arr, arr)
    else:
        res = Send(arr, 0, tag=rank)
        assert res == 0
        assert np.array_equal(_arr, arr)


def test_send_recv_jit():
    from mpi4jax import Send, Recv

    arr = np.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(lambda x: Recv(x, source=proc, tag=proc))(arr)
            assert np.array_equal(res, np.ones_like(arr) * proc)
            assert np.array_equal(_arr, arr)
    else:
        res = jax.jit(lambda x: Send(x, 0, tag=rank))(arr)
        assert res == 0
        assert np.array_equal(_arr, arr)


def test_send_recv_vmap():
    from mpi4jax import Send, Recv

    arr = np.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            res = jax.vmap(lambda x: Recv(x, source=proc, tag=proc), in_axes=0)(arr)
            assert np.array_equal(res, np.ones_like(arr) * proc)
            assert np.array_equal(_arr, arr)
    else:
        res = jax.vmap(lambda x: Send(x, 0, tag=rank), in_axes=0)(arr)
        assert res == 0
        assert np.array_equal(_arr, arr)


def test_send_recv_jit_vmap():
    from mpi4jax import Send, Recv

    arr = np.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(jax.vmap(lambda x: Recv(x, source=proc, tag=proc), in_axes=0))(arr)
            assert np.array_equal(res, np.ones_like(arr) * proc)
            assert np.array_equal(_arr, arr)
    else:
        res = jax.jit(jax.vmap(lambda x: Send(x, 0, tag=rank), in_axes=0))(arr)
        assert res == 0
        assert np.array_equal(_arr, arr)


def test_send_recv_status():
    from mpi4jax import Send, Recv

    arr = np.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res = jax.jit(lambda x: Recv(x, source=proc, tag=proc, status=status))(arr)
            assert np.array_equal(res, np.ones_like(arr) * proc)
            assert np.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        res = jax.jit(lambda x: Send(x, 0, tag=rank))(arr)
        assert res == 0
        assert np.array_equal(_arr, arr)
