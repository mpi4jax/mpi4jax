from mpi4py import MPI
import jax
from jax.scipy.sparse.linalg import cg
from mpi4jax._src.jax_compat import versiontuple
import pytest


def test_custom_linear_solver():
    import mpi4jax

    if versiontuple(jax.__version__) < versiontuple("0.5.1"):
        # see jax-ml/jax#26087
        pytest.xfail("JAX<0.5.1 does not support linear solves in no-token mode")

    k = jax.random.key(1)
    b = jax.random.normal(k, (24,))

    def mat_vec(v):
        res, _ = mpi4jax.allreduce(v, op=MPI.SUM, comm=MPI.COMM_WORLD)
        return res

    Aop = jax.tree_util.Partial(mat_vec)

    x, info = cg(Aop, b)
    assert jax.numpy.allclose(MPI.COMM_WORLD.size * x, b)

    x, info = jax.jit(cg)(Aop, b)
    assert jax.numpy.allclose(MPI.COMM_WORLD.size * x, b)
