import pytest

import importlib


@pytest.mark.parametrize(
    "fake_jaxlib_version",
    ["0.0.0", "0.1.61", "0.1.61+cuda110", "0.1.61a1", "0.0.0.0", "0.1.60.dev"],
)
def test_raise_on_outdated_jaxlib(fake_jaxlib_version, monkeypatch):
    import mpi4jax
    import jaxlib

    with monkeypatch.context() as m:
        m.setattr(jaxlib, "__version__", fake_jaxlib_version)

        with pytest.raises(RuntimeError) as excinfo:
            importlib.reload(mpi4jax._src)

        assert "mpi4jax requires" in str(excinfo.value)


@pytest.mark.parametrize(
    "fake_jaxlib_version",
    ["0.1.62", "1.1.0", "0.1.62+cuda110", "0.2.0rc1", "1.1.1.1", "0.1.63.dev"],
)
def test_no_raise_on_current_jaxlib(fake_jaxlib_version, monkeypatch):
    import mpi4jax
    import jaxlib

    with monkeypatch.context() as m:
        m.setattr(jaxlib, "__version__", fake_jaxlib_version)
        importlib.reload(mpi4jax._src)
