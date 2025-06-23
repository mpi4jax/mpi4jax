from mpi4py import MPI
import jax
from jax.scipy.sparse.linalg import cg


def test_custom_linear_solver():
    import mpi4jax

    k = jax.random.key(1)
    b = jax.random.normal(k, (24,))

    def mat_vec(v):
        res = mpi4jax.allreduce(v, op=MPI.SUM, comm=MPI.COMM_WORLD)
        return res

    Aop = jax.tree_util.Partial(mat_vec)

    x, info = cg(Aop, b)
    assert jax.numpy.allclose(MPI.COMM_WORLD.size * x, b)

    x, info = jax.jit(cg)(Aop, b)
    assert jax.numpy.allclose(MPI.COMM_WORLD.size * x, b)
