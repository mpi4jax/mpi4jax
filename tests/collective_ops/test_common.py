import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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

    # passing a mostly empty env seems to be the only way to
    # force MPI to initialize again
    env = dict(
        HOME=os.getenv("HOME", ""),
        PATH=os.getenv("PATH", ""),
        PYTHONPATH=os.getenv("PYTHONPATH", ""),
        COVERAGE_PROCESS_START="pyproject.toml",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        LD_LIBRARY_PATH=os.getenv("LD_LIBRARY_PATH", ""),
    )

    # non-standard Intel MPI env var for libfabric.
    if "FI_PROVIDER_PATH" in os.environ:
        env["FI_PROVIDER_PATH"] = os.getenv("FI_PROVIDER_PATH")

    proc = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        bufsize=0,
        timeout=timeout,
        text=True,
        env=env,
    )
    return proc


@pytest.mark.skipif(rank > 0, reason="Runs only on rank 0")
def test_abort_on_error(tmp_path):
    from textwrap import dedent

    # test in subprocess so we don't kill the testing process itself
    test_script = dedent(
        """
        import jax
        import jax.numpy as jnp

        from mpi4py import MPI
        from mpi4jax import send

        comm = MPI.COMM_WORLD
        assert comm.Get_size() == 1

        # send to non-existing rank
        @jax.jit
        def send_jit(x):
            send(x, dest=100, comm=comm)

        send_jit(jnp.ones(10))
    """
    )

    proc = run_in_subprocess(test_script, tmp_path / "abort.py")
    print(proc.stderr)
    assert proc.returncode != 0
    assert "r0 | MPI_Send returned error code" in proc.stderr


@pytest.mark.skipif(rank > 0, reason="Runs only on rank 0")
def test_deadlock_on_exit(tmp_path):
    from textwrap import dedent

    # without our custom atexit handler, this would deadlock most of the time
    test_script = dedent(
        """
        import jax
        import jax.numpy as jnp

        from mpi4py import MPI
        from mpi4jax import sendrecv

        comm = MPI.COMM_WORLD
        assert comm.Get_size() == 1

        # sendrecv to self
        jax.jit(
            lambda x: sendrecv(sendbuf=x, recvbuf=x, source=0, dest=0, comm=comm)[0]
        )(jnp.ones(10))
    """
    )

    proc = run_in_subprocess(test_script, tmp_path / "deadlock_on_exit.py")
    assert proc.returncode == 0, proc.stderr


def test_debug_logging(capsys):
    import re
    from mpi4jax import allreduce
    from mpi4jax._src.xla_bridge.mpi_xla_bridge import set_logging

    arr = jnp.ones((3, 2))

    set_logging(True)
    res = jax.jit(lambda x: allreduce(x, op=MPI.SUM)[0])(arr)
    res.block_until_ready()

    captured = capsys.readouterr().out
    start_msg, end_msg, _ = captured.split("\n")
    assert re.match(
        rf"r{rank} \| \w{{8}} \| MPI_Allreduce with {arr.size} items", start_msg
    )
    assert re.match(
        rf"r{rank} \| \w{{8}} \| MPI_Allreduce done with code 0 \(\d\.\d{{2}}e[+-]?\d+s\)",
        end_msg,
    )

    set_logging(False)
    res = jax.jit(lambda x: allreduce(x, op=MPI.SUM)[0])(arr)
    res.block_until_ready()

    captured = capsys.readouterr().out
    assert not captured


def test_set_logging_from_envvar():
    import os
    import importlib

    from mpi4jax._src import xla_bridge
    from mpi4jax._src.xla_bridge.mpi_xla_bridge import set_logging, get_logging

    os.environ["MPI4JAX_DEBUG"] = "1"
    importlib.reload(xla_bridge)
    assert get_logging()

    os.environ["MPI4JAX_DEBUG"] = "0"
    importlib.reload(xla_bridge)
    assert not get_logging()

    set_logging(True)
    assert get_logging()

    del os.environ["MPI4JAX_DEBUG"]
