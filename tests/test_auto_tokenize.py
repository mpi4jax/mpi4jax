from mpi4py import MPI

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def test_allreduce():
    from mpi4jax import allreduce, auto_tokenize

    arr = jnp.ones((3, 2))

    def f(a):
        res, _ = allreduce(a, op=MPI.SUM)
        res, _ = allreduce(res, op=MPI.SUM)
        return res
    res = auto_tokenize(f)(arr)
    np.testing.assert_allclose(res, arr * size ** 2)

def test_nested_jits():
    from mpi4jax import allreduce, auto_tokenize

    arr = jnp.ones((3, 2))
    @jax.jit
    def f(a):
        res, _ = allreduce(a, op=MPI.SUM)
        @jax.jit
        def g(b):
            res, _ = allreduce(b, op=MPI.SUM)
            res, _ = allreduce(res, op=MPI.SUM)
            return res
        return g(res)

    res = auto_tokenize(f)(arr)
    np.testing.assert_allclose(res, arr * size ** 3)
