#!/usr/bin/env python

"""
Run with

$ mpirun -n <nproc> python test.py
"""

import jax
import jax.numpy as np
import pytest

from mpi4py import MPI

print(MPI.get_vendor())

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

print("MPI rank = ", rank)
print("MPI size = ", size)

from mpi4jax import Allreduce


def test_allreduce():
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
