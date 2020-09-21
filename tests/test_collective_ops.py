#!/usr/bin/env python

"""
Run with

$ mpirun -n <nproc> python -m pytest .
"""

import jax
import jax.config
import jax.numpy as jnp
import pytest

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

    res, token = jax.jit(lambda x: Allreduce(x, op=MPI.SUM))(arr)
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

    res, token = jax.jit(lambda x: Allreduce(x, op=MPI.SUM))(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_grad():
    from mpi4jax import Allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, grad = jax.value_and_grad(lambda x: Allreduce(x, op=MPI.SUM)[0].sum())(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, arr)

    res, grad = jax.jit(
        jax.value_and_grad(lambda x: Allreduce(x, op=MPI.SUM)[0].sum())
    )(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, arr)

    def testfun(x):
        y, token = Allreduce(x, op=MPI.SUM)
        z = x + 2 * y  # noqa: F841
        res, token2 = Allreduce(x, op=MPI.SUM, token=token)
        return res.sum()

    res, grad = jax.jit(jax.value_and_grad(testfun))(arr)
    assert jnp.array_equal(res, arr.sum() * size)
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

    if rank == 0:
        for proc in range(1, size):
            res, token = jax.jit(lambda x: Recv(x, source=proc, tag=proc))(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        jax.jit(lambda x: Send(x, 0, tag=rank))(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_jit():
    from mpi4jax import Recv, Send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            res, token = jax.jit(lambda x: Recv(x, source=proc, tag=proc))(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        jax.jit(lambda x: Send(x, 0, tag=rank))(arr)
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

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res, token = jax.jit(
                lambda x: Recv(x, source=proc, tag=proc, status=status)
            )(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        jax.jit(lambda x: Send(x, 0, tag=rank))(arr)
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
    res, token = jax.jit(
        lambda x, y: Sendrecv(x, y, source=other, dest=other, status=status)
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

    res, token = jax.jit(lambda x, y: Sendrecv(x, y, source=other, dest=other))(
        arr, arr
    )

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_scalar_jit():
    from mpi4jax import Sendrecv

    arr = 1 * rank
    _arr = arr

    other = 1 - rank

    res, token = jax.jit(lambda x, y: Sendrecv(x, y, source=other, dest=other))(
        arr, arr
    )

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
        jax.jit(lambda x: Send(x, dest=100, comm=comm))(
            jnp.ones(10)
        )
    """
    )

    proc = run_in_subprocess(test_script, tmp_path / "abort.py")
    assert proc.returncode != 0
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
        jax.jit(lambda x: Sendrecv(sendbuf=x, recvbuf=x, source=0, dest=0, comm=comm))(
            jnp.ones(10)
        )
    """
    )

    proc = run_in_subprocess(test_script, tmp_path / "deadlock_on_exit.py")
    assert proc.returncode == 0, proc.stderr


def test_debug_logging_disabled(capsys, monkeypatch):
    from mpi4jax import Allreduce
    from mpi4jax.cython.mpi_xla_bridge import set_logging

    arr = jnp.ones((3, 2))

    set_logging(True)
    set_logging(False)

    res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM))(arr)
    res[0].block_until_ready()

    captured = capsys.readouterr()
    assert not captured.out


def test_debug_logging_enabled(capsys, monkeypatch):
    from mpi4jax import Allreduce
    from mpi4jax.cython.mpi_xla_bridge import set_logging

    arr = jnp.ones((3, 2))
    try:
        set_logging(True)
        res = jax.jit(lambda x: Allreduce(x, op=MPI.SUM))(arr)
        res[0].block_until_ready()
    finally:
        set_logging(False)

    captured = capsys.readouterr()
    assert captured.out.startswith(f"r{rank} | MPI_Allreduce with token")
