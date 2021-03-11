import pytest

import jax
import jax.numpy as jnp

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_allreduce_jit_deprecated():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    with pytest.warns(UserWarning, match="deprecated"):
        res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM)[0])(arr)

    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_bcast_jit_deprecated():
    from mpi4jax import Bcast

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    if rank != 0:
        _arr = _arr * 0

    with pytest.warns(UserWarning, match="deprecated"):
        res = jax.jit(lambda x: Bcast(x, root=0)[0])(arr)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_jit_deprecated():
    from mpi4jax import Recv, Send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    @jax.jit
    def send_jit_deprecated(x):
        Send(x, 0, tag=rank)
        return x

    with pytest.warns(UserWarning, match="deprecated"):
        if rank == 0:
            for proc in range(1, size):
                res = jax.jit(lambda x: Recv(x, source=proc, tag=proc)[0])(arr)
                assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
                assert jnp.array_equal(_arr, arr)
        else:
            send_jit_deprecated(arr)
            assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jit_deprecated():
    from mpi4jax import Sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    with pytest.warns(UserWarning, match="deprecated"):
        res = jax.jit(lambda x, y: Sendrecv(x, y, source=other, dest=other)[0])(
            arr, arr
        )

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
