#!/usr/bin/env python

"""
Run with

$ mpirun -n <nproc> python -m pytest .
"""

import pytest

# don't hog memory if running on GPU
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax  # noqa: E402
import jax.config  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.enable_omnistaging()

from mpi4py import MPI  # noqa: E402

print(MPI.get_vendor())

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

print("MPI rank = ", rank)
print("MPI size = ", size)


def test_allreduce():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, token = Allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_jit():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM)[0])(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_scalar():
    from mpi4jax import Allreduce

    arr = 1
    _arr = 1

    res, token = Allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_scalar_jit():
    from mpi4jax import Allreduce

    arr = 1
    _arr = 1

    res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM)[0])(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduceT():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, _ = Allreduce(arr, op=MPI.SUM, _transpose=True)
    assert jnp.array_equal(_arr, res)


def test_allreduceT_jit():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, _ = jax.jit(lambda x: Allreduce(x, op=MPI.SUM, _transpose=True))(arr)
    assert jnp.array_equal(_arr, res)


def test_allreduce_transpose():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    (res,) = jax.linear_transpose(lambda x: Allreduce(x, op=MPI.SUM)[0], arr)(_arr)
    assert jnp.array_equal(_arr, res)


def test_allreduce_transpose_jit():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    def f(x):
        (res,) = jax.linear_transpose(lambda x: Allreduce(x, op=MPI.SUM)[0], arr)(x)
        return res

    res = jax.jit(f)(arr)
    assert jnp.array_equal(_arr, res)


def test_allreduce_transpose2():
    # test transposing twice
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()
    _arr2 = arr.copy()

    def lt(y):
        return jax.linear_transpose(lambda x: Allreduce(x, op=MPI.SUM)[0], arr)(y)[0]

    (res,) = jax.linear_transpose(lt, _arr)(_arr2)
    expected, _ = Allreduce(_arr2, op=MPI.SUM)
    assert jnp.array_equal(expected, res)


def test_allreduce_transpose2_jit():
    # test transposing twice
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()
    _arr2 = arr.copy()

    def lt(y):
        return jax.linear_transpose(lambda x: Allreduce(x, op=MPI.SUM)[0], arr)(y)[0]

    def f(x):
        (res,) = jax.linear_transpose(lt, _arr)(x)
        return res

    res = jax.jit(f)(_arr2)
    expected, _ = Allreduce(_arr2, op=MPI.SUM)
    assert jnp.array_equal(expected, res)


def test_allreduceT_transpose():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    (res,) = jax.linear_transpose(
        lambda x: Allreduce(x, op=MPI.SUM, _transpose=True)[0], arr
    )(_arr)
    expected, _ = Allreduce(_arr, op=MPI.SUM)
    assert jnp.array_equal(expected, res)


def test_allreduceT_transpose_jit():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    def f(x):
        (res,) = jax.linear_transpose(
            lambda x: Allreduce(x, op=MPI.SUM, _transpose=True)[0], arr
        )(x)
        return res

    res = jax.jit(f)(_arr)
    expected, _ = Allreduce(_arr, op=MPI.SUM)
    assert jnp.array_equal(expected, res)


def test_allreduceT_transpose2():
    # test transposing twice
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()
    _arr2 = arr.copy()

    def lt(y):
        return jax.linear_transpose(
            lambda x: Allreduce(x, op=MPI.SUM, _transpose=True)[0], arr
        )(y)[0]

    (res,) = jax.linear_transpose(lt, _arr)(_arr2)
    assert jnp.array_equal(_arr2, res)


def test_allreduceT_transpose2_jit():
    # test transposing twice
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()
    _arr2 = arr.copy()

    def lt(y):
        return jax.linear_transpose(
            lambda x: Allreduce(x, op=MPI.SUM, _transpose=True)[0], arr
        )(y)[0]

    def f(x):
        (res,) = jax.jit(jax.linear_transpose(lt, _arr))(x)
        return res

    res = jax.jit(f)(_arr2)
    assert jnp.array_equal(_arr2, res)


def test_allreduce_grad():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, grad = jax.value_and_grad(lambda x: Allreduce(x, op=MPI.SUM)[0].sum())(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)

    res, grad = jax.jit(
        jax.value_and_grad(lambda x: Allreduce(x, op=MPI.SUM)[0].sum())
    )(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)

    def testfun(x):
        y, token = Allreduce(x, op=MPI.SUM)
        z = x + 2 * y  # noqa: F841
        res, token2 = Allreduce(x, op=MPI.SUM)
        return res.sum()

    res, grad = jax.jit(jax.value_and_grad(testfun))(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)


def test_allreduce_jvp():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, jvp = jax.jvp(lambda x: Allreduce(x, op=MPI.SUM)[0], (arr,), (_arr,))

    expected, _ = Allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(expected, res)
    expected, _ = Allreduce(_arr, op=MPI.SUM)
    assert jnp.array_equal(expected, jvp)


def test_allreduce_vjp():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, vjp_fun = jax.vjp(lambda x: Allreduce(x, op=MPI.SUM)[0], arr)
    (vjp,) = vjp_fun(_arr)

    expected, _ = Allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(expected, res)
    assert jnp.array_equal(_arr, vjp)


def test_allreduceT_jvp():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, jvp = jax.jvp(
        lambda x: Allreduce(x, op=MPI.SUM, _transpose=True)[0],
        (arr,),
        (_arr,),
    )
    assert jnp.array_equal(arr, res)
    assert jnp.array_equal(_arr, jvp)


def test_allreduceT_vjp():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, vjp_fun = jax.vjp(lambda x: Allreduce(x, op=MPI.SUM, _transpose=True)[0], arr)
    (vjp,) = vjp_fun(_arr)

    assert jnp.array_equal(arr, res)
    expected, _ = Allreduce(_arr, op=MPI.SUM)
    assert jnp.array_equal(expected, vjp)


def test_bcast():
    from mpi4jax import Bcast

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    if rank != 0:
        _arr = _arr * 0

    res, token = Bcast(_arr, root=0)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


def test_bcast_jit():
    from mpi4jax import Bcast

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    if rank != 0:
        _arr = _arr * 0

    res = jax.jit(lambda x: Bcast(x, root=0)[0])(arr)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


def test_bcast_scalar():
    from mpi4jax import Bcast

    arr = 1
    _arr = 1

    if rank != 0:
        _arr = _arr * 0

    res, token = Bcast(_arr, root=0)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


def test_bcast_scalar_jit():
    from mpi4jax import Bcast

    arr = 1
    _arr = 1

    if rank != 0:
        _arr = _arr * 0

    res = jax.jit(lambda x: Bcast(x, root=0)[0])(_arr)
    assert jnp.array_equal(res, arr)
    if rank == 0:
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv():
    from mpi4jax import Recv, Send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            res, token = Recv(arr, source=proc, tag=proc)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        Send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_scalar():
    from mpi4jax import Recv, Send

    arr = 1 * rank
    _arr = 1 * rank

    if rank == 0:
        for proc in range(1, size):
            res, token = Recv(arr, source=proc, tag=proc)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        Send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_scalar_jit():
    from mpi4jax import Recv, Send

    arr = 1 * rank
    _arr = 1 * rank

    @jax.jit
    def send_jit(x):
        Send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(lambda x: Recv(x, source=proc, tag=proc)[0])(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_jit():
    from mpi4jax import Recv, Send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    @jax.jit
    def send_jit(x):
        Send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(lambda x: Recv(x, source=proc, tag=proc)[0])(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_send_recv_deadlock():
    from mpi4jax import Recv, Send

    # this deadlocks without proper token management
    @jax.jit
    def deadlock(arr):
        if rank == 0:
            # send, then receive
            token = Send(arr, 1)
            newarr, _ = Recv(arr, 1, token=token)
        else:
            # receive, then send
            newarr, token = Recv(arr, 0)
            Send(arr, 0, token=token)
        return newarr

    arr = jnp.ones(10) * rank
    arr = deadlock(arr)
    assert jnp.array_equal(arr, jnp.ones_like(arr) * (1 - rank))


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_status():
    from mpi4jax import Recv, Send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res, token = Recv(arr, source=proc, tag=proc, status=status)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        Send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_status_jit():
    from mpi4jax import Recv, Send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    @jax.jit
    def send_jit(x):
        Send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res = jax.jit(lambda x: Recv(x, source=proc, tag=proc, status=status)[0])(
                arr
            )
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv():
    from mpi4jax import Sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    res, token = Sendrecv(arr, arr, source=other, dest=other)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_status():
    from mpi4jax import Sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    status = MPI.Status()
    res, token = Sendrecv(arr, arr, source=other, dest=other, status=status)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
    assert status.Get_source() == other


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_status_jit():
    from mpi4jax import Sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    status = MPI.Status()
    res = jax.jit(
        lambda x, y: Sendrecv(x, y, source=other, dest=other, status=status)[0]
    )(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
    assert status.Get_source() == other


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_scalar():
    from mpi4jax import Sendrecv

    arr = 1 * rank
    _arr = arr

    other = 1 - rank

    res, token = Sendrecv(arr, arr, source=other, dest=other)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jit():
    from mpi4jax import Sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    other = 1 - rank

    res = jax.jit(lambda x, y: Sendrecv(x, y, source=other, dest=other)[0])(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_scalar_jit():
    from mpi4jax import Sendrecv

    arr = 1 * rank
    _arr = arr

    other = 1 - rank

    res = jax.jit(lambda x, y: Sendrecv(x, y, source=other, dest=other)[0])(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


def run_in_subprocess(code, test_file, timeout=10):
    """Runs given code string in a subprocess"""
    import os
    import subprocess
    import sys
    from textwrap import dedent

    # this enables us to measure coverage in subprocesses
    cov_preamble = dedent(
        """
    try:
        import coverage
        coverage.process_startup()
    except ImportError:
        pass

    """
    )

    test_file.write_text(cov_preamble + code)

    proc = subprocess.run(
        [sys.executable, test_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        timeout=timeout,
        universal_newlines=True,
        # passing a mostly empty env seems to be the only way to
        # force MPI to initialize again
        env=dict(
            PATH=os.environ["PATH"],
            COVERAGE_PROCESS_START="pyproject.toml",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
        ),
    )
    return proc


@pytest.mark.skipif(rank > 0, reason="Runs only on rank 0")
def test_abort_on_error(tmp_path):
    from textwrap import dedent

    # test in subprocess so we don't kill the testing process itself
    test_script = dedent(
        """
        import jax
        jax.config.enable_omnistaging()
        import jax.numpy as jnp

        from mpi4py import MPI
        from mpi4jax import Send

        comm = MPI.COMM_WORLD
        assert comm.Get_size() == 1

        # send to non-existing rank
        @jax.jit
        def send_jit(x):
            Send(x, dest=100, comm=comm)

        send_jit(jnp.ones(10))
    """
    )

    proc = run_in_subprocess(test_script, tmp_path / "abort.py")
    assert proc.returncode != 0
    print(proc.stderr)
    assert "r0 | MPI_Send returned error code" in proc.stderr


@pytest.mark.skipif(rank > 0, reason="Runs only on rank 0")
def test_deadlock_on_exit(tmp_path):
    from textwrap import dedent

    # without our custom atexit handler, this would deadlock most of the time
    test_script = dedent(
        """
        import jax
        jax.config.enable_omnistaging()
        import jax.numpy as jnp

        from mpi4py import MPI
        from mpi4jax import Sendrecv

        comm = MPI.COMM_WORLD
        assert comm.Get_size() == 1

        # sendrecv to self
        jax.jit(
            lambda x: Sendrecv(sendbuf=x, recvbuf=x, source=0, dest=0, comm=comm)[0]
        )(jnp.ones(10))
    """
    )

    proc = run_in_subprocess(test_script, tmp_path / "deadlock_on_exit.py")
    assert proc.returncode == 0, proc.stderr


def test_set_debug_logging(capsys):
    from mpi4jax import Allreduce
    from mpi4jax.cython.mpi_xla_bridge import set_logging

    arr = jnp.ones((3, 2))
    set_logging(True)
    res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM)[0])(arr)
    res.block_until_ready()

    captured = capsys.readouterr()
    assert captured.out.startswith(f"r{rank} | MPI_Allreduce with token")

    set_logging(False)
    res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM)[0])(arr)
    res.block_until_ready()

    captured = capsys.readouterr()
    assert not captured.out
