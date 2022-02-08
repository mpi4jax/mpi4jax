import pytest

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

        res = g(res)
        res = g(res)
        return res

    res = auto_tokenize(f)(arr)
    np.testing.assert_allclose(res, arr * size ** 5)


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

        res = g(res)
        res = g(res)
        return res

    res = auto_tokenize(f)(arr)
    np.testing.assert_allclose(res, arr * size ** 5)


@pytest.mark.skipif(size < 2, reason="need 2 processes to test send/recv")
def test_send_recv_tokenizer():
    from mpi4jax import recv, send, auto_tokenize

    def simple_message_pass(arr):
        # Here, we test a crazy send/recv pattern that is extremely likely to deadlock unless
        # the order is preserved.
        if rank == 0:
            b, _ = recv(arr, source=1, comm=comm)
            return b
        elif rank == 1:
            a = arr + 1
            send(a, dest=0, comm=comm)
            return arr
        return jnp.zeros_like(arr)

    res = auto_tokenize(simple_message_pass)(jnp.zeros((2, 2)))
    if rank == 0:
        np.testing.assert_allclose(res, jnp.ones((2, 2)))
    if rank == 1:
        np.testing.assert_allclose(res, jnp.zeros((2, 2)))

@pytest.mark.skipif(size < 2, reason="need 2 processes to test send/recv")
def test_send_recv_hotpotato_tokenizer():
    from mpi4jax import recv, send, auto_tokenize

    def hot_potato(arr):
        # Here, we test a crazy send/recv pattern that is extremely likely to deadlock unless
        # the order is preserved.
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
            send(a + b , dest=0, comm=comm)
            send(a * b, dest=0, comm=comm)
            c, _ = recv(arr, source=0, comm=comm)
            send(b - c, dest=0, comm=comm)
            d, _ = recv(arr, source=0, comm=comm)
            e, _ = recv(arr, source=0, comm=comm)
            return d + e
        return jnp.zeros_like(arr)

    res = auto_tokenize(hot_potato)(jnp.zeros((2, 2)))
    if rank == 0:
        np.testing.assert_allclose(res, jnp.ones((2, 2)) * -4)
    if rank == 1:
        np.testing.assert_allclose(res, jnp.ones((2, 2)) * 11)