def test_flush():
    from mpi4jax import has_hip_support

    assert isinstance(has_hip_support(), bool)
