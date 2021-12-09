def test_flush():
    from mpi4jax import has_cuda_support

    assert isinstance(has_cuda_support(), bool)
