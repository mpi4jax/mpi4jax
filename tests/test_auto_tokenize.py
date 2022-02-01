from mpi4py import MPI

import jax
import jax.numpy as jnp
import numpy as np

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
    np.testing.assert_allclose(res, arr * size * 2)

