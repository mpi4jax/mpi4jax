import pytest

import importlib


def test_raise_on_outdated_jaxlib(monkeypatch):
    import mpi4jax
    import jaxlib

    with monkeypatch.context() as m:
        m.setattr(jaxlib, "__version__", "0.0.0")

        with pytest.raises(RuntimeError) as excinfo:
            importlib.reload(mpi4jax._src)

        assert "mpi4jax requires" in str(excinfo.value)
