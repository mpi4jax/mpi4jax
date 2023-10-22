import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp
import numpy as np

from mpi4jax._src.utils import ordered_effect

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# TODO: remove once we require JAX>=0.4.16
pytestmark = pytest.mark.skipif(
    ordered_effect is None, reason="ordered_effect not implemented in this version"
)


def test_allreduce():
    from mpi4jax.experimental.notoken import allreduce

    arr = jnp.ones((3, 2))

    def f(a):
        res = allreduce(a, op=MPI.SUM)
        res = allreduce(res, op=MPI.SUM)
        return res

    res = f(arr)
    np.testing.assert_allclose(res, arr * size**2)


def test_nested_jits():
    from mpi4jax.experimental.notoken import allreduce

    arr = jnp.ones((3, 2))

    @jax.jit
    def f(a):
        res = allreduce(a, op=MPI.SUM)

        @jax.jit
        def g(b):
            res = allreduce(b, op=MPI.SUM)
            res = allreduce(res, op=MPI.SUM)
            return res

        res = g(res)
        res = g(res)
        return res

    res = f(arr)
    np.testing.assert_allclose(res, arr * size**5)


@pytest.mark.skipif(size < 2, reason="need 2 processes")
def test_send_recv_tokenizer():
    from mpi4jax.experimental.notoken import recv, send

    def simple_message_pass(arr):
        if rank == 0:
            b = recv(arr, source=1, comm=comm)
            return b
        elif rank == 1:
            a = arr + 1
            send(a, dest=0, comm=comm)
            return arr
        return jnp.zeros_like(arr)

    res = jax.jit(simple_message_pass)(jnp.zeros((2, 2)))
    if rank == 0:
        np.testing.assert_allclose(res, jnp.ones((2, 2)))
    if rank == 1:
        np.testing.assert_allclose(res, jnp.zeros((2, 2)))


@pytest.mark.skipif(size < 2, reason="need 2 processes")
def test_send_recv_hotpotato_tokenizer():
    from mpi4jax.experimental.notoken import recv, send, barrier

    def hot_potato(arr):
        # Here, we test a send/recv pattern that is extremely likely to return the
        # wrong result unless the order is preserved.
        barrier()  # Free barrier test aswell.
        if rank == 0:
            a = arr + 1
            b = recv(arr, source=1, comm=comm)
            send(a, dest=1, comm=comm)
            c = recv(arr, source=1, comm=comm)
            d = recv(arr, source=1, comm=comm)
            e = c + d
            send(e, dest=1, comm=comm)
            f = recv(arr, source=1, comm=comm)
            send(e + f, dest=1, comm=comm)
            send(e * d, dest=1, comm=comm)
            return f
        elif rank == 1:
            a = arr + 2
            send(a, dest=0, comm=comm)
            b = recv(arr, source=0, comm=comm)
            send(a + b, dest=0, comm=comm)
            send(a * b, dest=0, comm=comm)
            c = recv(arr, source=0, comm=comm)
            send(b - c, dest=0, comm=comm)
            d = recv(arr, source=0, comm=comm)
            e = recv(arr, source=0, comm=comm)
            return d + e
        return jnp.zeros_like(arr)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ! Proof that this test actually works !
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    # Uncomment the below line of code.
    # The final result will be wrong and the asserts will fail.
    #
    # ---------------------------
    # auto_tokenize = lambda x: x
    # ---------------------------

    real_result = hot_potato(jnp.zeros((2, 2)))
    jitted_tokenized = jax.jit(hot_potato)(jnp.zeros((2, 2)))
    np.testing.assert_allclose(real_result, jitted_tokenized)
    # Assert to ourselves the results are correct.
    if rank == 0:
        np.testing.assert_allclose(jitted_tokenized, jnp.ones((2, 2)) * -4)
    if rank == 1:
        np.testing.assert_allclose(jitted_tokenized, jnp.ones((2, 2)) * 11)


def test_fori_loop_tokenizer():
    from mpi4jax.experimental.notoken import allreduce

    NUM_LOOPS = 6

    def sum_loop(i, args):
        (arr,) = args
        res = allreduce(arr, op=MPI.SUM)
        return [res]

    def my_method(arr):
        return jax.lax.fori_loop(0, NUM_LOOPS, sum_loop, [arr])

    res = jax.jit(my_method)(jnp.ones((2, 2)))
    np.testing.assert_allclose(res[0], np.ones((2, 2)) * size**NUM_LOOPS)


def test_while_loop_tokenizer():
    from mpi4jax.experimental.notoken import allreduce

    def sum_loop(arr):
        res = allreduce(arr, op=MPI.SUM)
        res = res + 1
        return res

    def cond(arr):
        return jnp.all(arr < 1000)

    def my_method(arr):
        return jax.lax.while_loop(cond, sum_loop, arr)

    res = jax.jit(my_method)(jnp.ones((2, 2)))
    assert (res >= np.ones((2, 2)) * 1000).all()


def test_cond_tokenizer():
    from mpi4jax.experimental.notoken import allreduce

    def branch1(arr):
        res = allreduce(arr, op=MPI.PROD)
        return res

    def branch2(arr):
        res = allreduce(arr, op=MPI.SUM)
        return res

    def my_method(bool, arr):
        return jax.lax.cond(bool, branch1, branch2, arr)

    res1 = jax.jit(my_method)(
        jnp.asarray(True), jnp.ones((2, 2), dtype=jnp.int32)
    ).block_until_ready()
    res2 = jax.jit(my_method)(
        jnp.asarray(False), jnp.ones((2, 2), dtype=jnp.int32)
    ).block_until_ready()
    assert (res1 == 1).all()
    assert (res2 == size).all()


def test_allgather_scalar():
    from mpi4jax.experimental.notoken import allgather

    @jax.jit
    def f(arr):
        res = allgather(arr)
        return res

    res = f(jnp.asarray(rank))
    assert jnp.array_equal(res, jnp.arange(size))


def test_alltoall_jit():
    from mpi4jax.experimental.notoken import alltoall

    arr = jnp.ones((size, 3, 2)) * rank

    res = jax.jit(lambda x: alltoall(x))(arr)
    for p in range(size):
        assert jnp.array_equal(res[p], jnp.ones((3, 2)) * p)


def test_bcast_scalar_jit():
    from mpi4jax.experimental.notoken import bcast

    arr = 1
    _arr = 1

    if rank != 0:
        _arr = _arr * 0

    res = jax.jit(lambda x: bcast(x, root=0))(_arr)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


def test_gather_scalar_jit():
    from mpi4jax.experimental.notoken import gather

    arr = rank
    res = jax.jit(lambda x: gather(x, root=0))(arr)
    if rank == 0:
        assert jnp.array_equal(res, jnp.arange(size))
    else:
        assert jnp.array_equal(res, arr)


def test_reduce_scalar_jit():
    from mpi4jax.experimental.notoken import reduce

    arr = rank
    res = jax.jit(lambda x: reduce(x, op=MPI.SUM, root=0))(arr)
    if rank == 0:
        assert jnp.array_equal(res, sum(range(size)))
    else:
        assert jnp.array_equal(res, arr)


def test_scan_scalar_jit():
    from mpi4jax.experimental.notoken import scan

    arr = rank
    res = jax.jit(lambda x: scan(x, op=MPI.SUM))(arr)
    assert jnp.array_equal(res, sum(range(rank + 1)))


def test_scatter_jit():
    from mpi4jax.experimental.notoken import scatter

    if rank == 0:
        arr = jnp.stack([jnp.ones((3, 2)) * r for r in range(size)], axis=0)
    else:
        arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: scatter(x, root=0))(arr)
    assert jnp.array_equal(res, jnp.ones((3, 2)) * rank)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_status_jit():
    from mpi4jax.experimental.notoken import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    status = MPI.Status()
    res = jax.jit(lambda x, y: sendrecv(x, y, source=other, dest=other, status=status))(
        arr, arr
    )

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
    assert status.Get_source() == other


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_while_loop_consistency():
    from mpi4jax.experimental.notoken import allreduce, sendrecv, barrier

    def cond(value):
        barrier()
        if rank == 1 or rank == 0:
            new_value = sendrecv(
                value, jnp.zeros_like(value), source=1 - rank, dest=1 - rank
            )
        return jnp.all(new_value < 10)

    def loop(value):
        barrier()
        new_value = allreduce(value, MPI.SUM)
        return new_value

    @jax.jit
    def run(value):
        return jax.lax.while_loop(cond, loop, value)

    res = run(1.0)
    assert res >= 10


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_cond_consistency():
    from mpi4jax.experimental.notoken import recv, send

    def hot_potato_right(arr):
        a = arr + 1
        _ = recv(arr, source=1, comm=comm)
        send(a, dest=1, comm=comm)
        c = recv(arr, source=1, comm=comm)
        d = recv(arr, source=1, comm=comm)
        e = c + d
        send(e, dest=1, comm=comm)
        f = recv(arr, source=1, comm=comm)
        send(e + f, dest=1, comm=comm)
        send(e * d, dest=1, comm=comm)
        return f

    def hot_potato_left(arr):
        a = arr + 2
        send(a, dest=0, comm=comm)
        b = recv(arr, source=0, comm=comm)
        send(a + b, dest=0, comm=comm)
        send(a * b, dest=0, comm=comm)
        c = recv(arr, source=0, comm=comm)
        send(b - c, dest=0, comm=comm)
        d = recv(arr, source=0, comm=comm)
        e = recv(arr, source=0, comm=comm)
        return d + e

    @jax.jit
    def run(arr, my_rank):
        res1 = jax.lax.cond(my_rank == 0, hot_potato_right, hot_potato_left, arr)
        return res1

    res = run(jnp.zeros((2, 2)), rank)

    if rank == 0:
        expected = -4
    else:
        expected = 11

    np.testing.assert_allclose(res, jnp.ones((2, 2)) * expected)
