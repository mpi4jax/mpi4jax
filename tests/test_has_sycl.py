def test_flush():
    from mpi4jax import has_sycl_support

    assert isinstance(has_sycl_support(), bool)
