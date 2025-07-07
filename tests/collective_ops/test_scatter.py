import pytest
from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_scatter():
    from mpi4jax import scatter

    if rank == 0:
        arr = jnp.stack([jnp.ones((3, 2)) * r for r in range(size)], axis=0)
    else:
        arr = jnp.ones((3, 2)) * rank

    res = scatter(arr, root=0)
    assert jnp.array_equal(res, jnp.ones((3, 2)) * rank)


def test_scatter_jit():
    from mpi4jax import scatter

    if rank == 0:
        arr = jnp.stack([jnp.ones((3, 2)) * r for r in range(size)], axis=0)
    else:
        arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: scatter(x, root=0))(arr)
    assert jnp.array_equal(res, jnp.ones((3, 2)) * rank)


@pytest.mark.skipif(rank > 0, reason="Only runs on root process")
def test_scatter_wrong_size():
    from mpi4jax import scatter

    arr = jnp.ones((size + 1,)) * rank
    with pytest.raises(ValueError) as excinfo:
        scatter(arr, root=0)

    assert "Scatter input must have shape" in str(excinfo.value)
