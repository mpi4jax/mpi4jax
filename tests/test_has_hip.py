def test_flush():
    from mpi4jax import has_rocm_support

    assert isinstance(has_rocm_support(), bool)
