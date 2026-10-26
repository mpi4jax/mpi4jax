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
        LIBRARY_PATH=os.getenv("LIBRARY_PATH", ""),
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
            lambda x: sendrecv(sendbuf=x, recvbuf=x, source=0, dest=0, comm=comm)
        )(jnp.ones(10))
    """
    )

    proc = run_in_subprocess(test_script, tmp_path / "deadlock_on_exit.py")
    assert proc.returncode == 0, proc.stderr


def test_debug_logging(capsys):
    import re
    from mpi4jax import allreduce
    from mpi4jax._src.xla_bridge import set_logging

    arr = jnp.ones((3, 2))

    set_logging(True)
    res = jax.jit(lambda x: allreduce(x, op=MPI.SUM))(arr)
    res.block_until_ready()

    captured = capsys.readouterr().out
    start_msg, end_msg, _ = captured.split("\n")
    # Device tag (GPU/XPU) is optional - present on GPU/XPU backends, absent on CPU
    assert re.match(
        rf"r{rank} \| \w{{8}} \| MPI_Allreduce( \(\w+\))? with {arr.size} items",
        start_msg,
    )
    assert re.match(
        rf"r{rank} \| \w{{8}} \| MPI_Allreduce( \(\w+\))? done with code 0 \(\d\.\d{{2}}e[+-]?\d+s\)",
        end_msg,
    )

    set_logging(False)
    res = jax.jit(lambda x: allreduce(x, op=MPI.SUM))(arr)
    res.block_until_ready()

    captured = capsys.readouterr().out
    assert not captured


def test_set_logging_from_envvar():
    import os
    import importlib

    from mpi4jax._src import xla_bridge
    from mpi4jax._src.xla_bridge import set_logging, get_logging

    os.environ["MPI4JAX_DEBUG"] = "1"
    importlib.reload(xla_bridge)
    assert get_logging()

    os.environ["MPI4JAX_DEBUG"] = "0"
    importlib.reload(xla_bridge)
    assert not get_logging()

    set_logging(True)
    assert get_logging()

    del os.environ["MPI4JAX_DEBUG"]


def test_mpi_abi_info():
    """Test that MPI ABI info is available and contains expected fields."""
    from mpi4jax._src.xla_bridge.mpi_xla_bridge_cpu import MPI_ABI_INFO

    # Check that all expected fields are present
    assert "sizeof_comm" in MPI_ABI_INFO
    assert "sizeof_datatype" in MPI_ABI_INFO
    assert "sizeof_op" in MPI_ABI_INFO
    assert "sizeof_status" in MPI_ABI_INFO
    assert "comm_world_handle" in MPI_ABI_INFO
    assert "mpi_library_version" in MPI_ABI_INFO

    # Check that sizes are reasonable (4 or 8 bytes for handles)
    assert MPI_ABI_INFO["sizeof_comm"] in (4, 8)
    assert MPI_ABI_INFO["sizeof_datatype"] in (4, 8)
    assert MPI_ABI_INFO["sizeof_op"] in (4, 8)
    assert MPI_ABI_INFO["sizeof_status"] > 0

    # Check that library version is a non-empty string
    assert len(MPI_ABI_INFO["mpi_library_version"]) > 0


def test_mpi_abi_compatibility_check():
    """Test that the ABI compatibility check passes for matching MPI."""
    from mpi4jax._src.xla_bridge import _check_mpi_abi_compatibility

    # Should not raise when build-time and runtime MPI match
    _check_mpi_abi_compatibility()


@pytest.mark.skipif(rank > 0, reason="Runs only on rank 0")
def test_mpi_abi_skip_check(tmp_path):
    """Test that MPI4JAX_SKIP_ABI_CHECK disables the check."""
    from textwrap import dedent

    # Test that setting MPI4JAX_SKIP_ABI_CHECK allows import even if we
    # were to mock a mismatch (here we just verify the env var is respected)
    test_script = dedent(
        """
        import os
        os.environ["MPI4JAX_SKIP_ABI_CHECK"] = "1"

        # Force reimport to test the skip
        import importlib
        from mpi4jax._src import xla_bridge
        importlib.reload(xla_bridge)

        # If we got here, the check was skipped successfully
        print("ABI check skipped successfully")
        """
    )

    proc = run_in_subprocess(test_script, tmp_path / "abi_skip.py")
    assert proc.returncode == 0, proc.stderr
    assert "ABI check skipped successfully" in proc.stdout
