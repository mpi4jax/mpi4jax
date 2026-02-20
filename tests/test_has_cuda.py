def test_has_cuda_support():
    from mpi4jax import has_cuda_support

    assert isinstance(has_cuda_support(), bool)
