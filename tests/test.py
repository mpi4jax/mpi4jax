#!/usr/bin/env python

"""
Run with

$ mpirun -n <nproc> python test.py
"""

import jax
import jax.numpy as np
from mpi4jax import Allreduce

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


def test_nojit(a):
    return Allreduce(a, op=MPI.SUM)


@jax.jit
def test_jit(a):
    return Allreduce(a, op=MPI.SUM)


def test_vmap_nojit(a):
    return jax.vmap(test_nojit, in_axes=0)(a)


def test_vmap_jit(a):
    return jax.jit(jax.vmap(test_nojit, in_axes=0))(a)


def test_grad_sum_nojit(a):
    return jax.value_and_grad(lambda x: Allreduce(x, op=MPI.SUM).sum())(arr)


def test_grad_sum_jit(a):
    return jax.jit(jax.value_and_grad(lambda x: Allreduce(x, op=MPI.SUM).sum()))(arr)


if __name__ == "__main__":
    arr = np.ones((3, 2))
    _arr = arr.copy()

    res = test_nojit(arr)
    if rank == 0:
        print("eval     : ", end="")
        print("✓ " if np.array_equal(res, arr * size) else "✗ ", end="")
        print("✓ " if np.array_equal(arr, _arr) else "✗ : modified input")

    res = test_jit(arr)
    if rank == 0:
        print("eval JIT : ", end="")
        print("✓ " if np.array_equal(res, arr * size) else "✗ ", end="")
        print("✓ " if np.array_equal(arr, _arr) else "✗: modified input")

    res = test_vmap_nojit(arr)
    if rank == 0:
        print("vmap     : ", end="")
        print("✓" if np.array_equal(res, arr * size) else "✗", end="")
        print(" ✓ " if np.array_equal(arr, _arr) else "✗: modified input")

    res = test_vmap_jit(arr)
    if rank == 0:
        print("vmap JIT : ", end="")
        print("✓" if np.array_equal(res, arr * size) else "✗", end="")
        print(" ✓ " if np.array_equal(arr, _arr) else "✗: modified input")

    res, grad = test_grad_sum_nojit(arr)
    if rank == 0:
        print("grad     : ", end="")
        print("✓ " if np.array_equal(res, arr.sum() * size) else "✗ ", end="")
        print("✓ " if np.array_equal(grad, np.ones(arr.shape)) else "✗ ", end="")
        print("✓ " if np.array_equal(arr, _arr) else "✗: modified input")

    res, grad = test_grad_sum_jit(arr)
    if rank == 0:
        print("grad JIT : ", end="")
        print("✓ " if np.array_equal(res, arr.sum() * size) else "✗ ", end="")
        print("✓ " if np.array_equal(grad, np.ones(arr.shape)) else "✗ ", end="")
        print("✓ " if np.array_equal(arr, _arr) else "✗: modified input")
