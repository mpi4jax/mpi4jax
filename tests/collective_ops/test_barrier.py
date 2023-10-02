import tempfile
import time
import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def cap_to_file(capsys, write_to):
    out = capsys.readouterr().out
    with open(write_to, "a") as f:
        f.write(out)


def test_barrier(capsys):
    """Verify that barrier blocks execution by printing messages before and after"""
    from mpi4jax._src.flush import flush
    from mpi4jax import barrier

    # pipe all messages to the same file
    tmpdir = tempfile.gettempdir()

    write_to = os.path.join(tmpdir, "mpi4jax-barrier.txt")
    if rank == 0:
        with open(write_to, "w"):
            pass

    print(f"r{rank} | start")

    time.sleep(rank * 0.2)
    cap_to_file(capsys, write_to)

    # without a barrier here, some ranks would start writing
    # "done" before everyone has writen "start"
    token = barrier()  # noqa: F841
    flush()

    print(f"r{rank} | done")

    time.sleep(rank * 0.2)
    cap_to_file(capsys, write_to)

    time.sleep(size * 0.2)

    with open(write_to) as f:
        outputs = f.readlines()

    assert len(outputs) == size * 2
    assert all(o.endswith("start\n") for o in outputs[:size])
    assert all(o.endswith("done\n") for o in outputs[size:])
