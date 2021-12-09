import pytest


def test_envvar_parsing():
    from mpi4jax._src.decorators import _is_falsy, _is_truthy

    assert _is_truthy("1")
    assert not _is_falsy("1")

    assert not _is_truthy("false")
    assert _is_falsy("false")

    assert not _is_truthy("foo")
    assert not _is_falsy("foo")


def test_missing_omnistaging(monkeypatch):
    import jax
    from mpi4jax._src.decorators import ensure_omnistaging

    with monkeypatch.context() as m:
        m.setattr(jax.config, "omnistaging_enabled", False)

        with pytest.raises(RuntimeError) as excinfo:
            ensure_omnistaging()

        assert "omnistaging" in str(excinfo.value)


def test_ensure_gpu_ext(monkeypatch):
    from mpi4jax._src import xla_bridge
    from mpi4jax._src.decorators import ensure_gpu_ext

    with monkeypatch.context() as m:
        m.setattr(xla_bridge, "HAS_GPU_EXT", False)

        with pytest.raises(ImportError) as excinfo:
            ensure_gpu_ext()

        assert "GPU extensions could not be imported" in str(excinfo.value)
