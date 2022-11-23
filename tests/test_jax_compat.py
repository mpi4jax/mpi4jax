import pytest

import warnings
import importlib


TEST_VERSIONS = {
    "0.0.0": (0, 0, 0),
    "0.1.61": (0, 1, 61),
    "0.1.61+cuda110": (0, 1, 61),
    "0.1.61a1": (0, 1, 61),
    "0.0.0.0": (0, 0, 0),
    "0.1.60.dev": (0, 1, 60),
}


@pytest.mark.parametrize(
    "fake_version",
    list(TEST_VERSIONS.keys()),
)
def test_versiontuple(fake_version):
    from mpi4jax._src.jax_compat import versiontuple

    assert versiontuple(fake_version) == TEST_VERSIONS[fake_version]


def test_version_warning(monkeypatch):
    import mpi4jax
    import jax

    # raise warning on too recent jax
    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", "99.99.99")
        m.setenv("MPI4JAX_NO_WARN_JAX_VERSION", "0")
        with pytest.warns(UserWarning) as w:
            importlib.reload(mpi4jax._src)
            assert "but you have 99.99.99" in str(w[0])

    # do not raise warning on outdated jax
    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", "00.00.00")
        m.setenv("MPI4JAX_NO_WARN_JAX_VERSION", "0")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            importlib.reload(mpi4jax._src)

    # do not raise when envvar is set
    with monkeypatch.context() as m:
        m.setattr(jax, "__version__", "99.99.99")
        m.setenv("MPI4JAX_NO_WARN_JAX_VERSION", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            importlib.reload(mpi4jax._src)
