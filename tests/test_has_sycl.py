def test_has_sycl_support():
    from mpi4jax import has_sycl_support

    assert isinstance(has_sycl_support(), bool)
