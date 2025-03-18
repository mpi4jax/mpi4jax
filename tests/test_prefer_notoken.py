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
