import pytest

import warnings


def test_prefer_notoken(monkeypatch):
    import mpi4jax
    import jax

    # If JAX < 0.5.0 we default to token mode if we not enforce it
    jax_ver = "0.4.38"
    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", jax_ver)
        m.delenv("MPI4JAX_PREFER_NOTOKEN", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mpi4jax._src.utils.prefer_notoken() is False

    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", jax_ver)
        m.setenv("MPI4JAX_PREFER_NOTOKEN", "0")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mpi4jax._src.utils.prefer_notoken() is False

    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", jax_ver)
        m.setenv("MPI4JAX_PREFER_NOTOKEN", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mpi4jax._src.utils.prefer_notoken() is True

    # Default to no token mode, if we not enforce it
    jax_ver = "0.5.0"
    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", jax_ver)
        m.delenv("MPI4JAX_PREFER_NOTOKEN", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mpi4jax._src.utils.prefer_notoken() is True

    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", jax_ver)
        m.setenv("MPI4JAX_PREFER_NOTOKEN", "0")
        with pytest.warns(UserWarning):
            assert mpi4jax._src.utils.prefer_notoken() is False

    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", jax_ver)
        m.setenv("MPI4JAX_PREFER_NOTOKEN", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mpi4jax._src.utils.prefer_notoken() is True


def test_custom_linalg_solver():
    import mpi4jax
    from mpi4py import MPI
    import jax
    from jax.scipy.sparse.linalg import cg

    if jax.__version__ == "0.5.0":
        # We need no-token mode for JAX>=0.5.0 but the code below
        # does not work for JAX<0.5.1 in no-token mode.
        return

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
