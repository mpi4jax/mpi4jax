import numpy as np

from mpi4py import MPI
from mpi4jax import recv, send, barrier

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def hot_potato(arr):
    # Here, we test a send/recv pattern that is extremely likely to return the
    # wrong result unless the order is preserved.
    barrier()  # Free barrier test aswell.
    if rank == 0:
        a = arr + 1
        b, _ = recv(arr, source=1, comm=comm)
        send(a, dest=1, comm=comm)
        c, _ = recv(arr, source=1, comm=comm)
        d, _ = recv(arr, source=1, comm=comm)
        e = c + d
        send(e, dest=1, comm=comm)
        f, _ = recv(arr, source=1, comm=comm)
        send(e + f, dest=1, comm=comm)
        send(e * d, dest=1, comm=comm)
        return f
    elif rank == 1:
        a = arr + 2
        send(a, dest=0, comm=comm)
        b, _ = recv(arr, source=0, comm=comm)
        send(a + b, dest=0, comm=comm)
        send(a * b, dest=0, comm=comm)
        c, _ = recv(arr, source=0, comm=comm)
        send(b - c, dest=0, comm=comm)
        d, _ = recv(arr, source=0, comm=comm)
        e, _ = recv(arr, source=0, comm=comm)
        return d + e
    return jnp.zeros_like(arr)


real_result = hot_potato(jnp.zeros((2, 2)))
jitted = jax.jit(jax.jit(hot_potato))(jnp.zeros((2, 2)))
np.testing.assert_allclose(real_result, jitted)
print("Success!")
