def test_flush():
    # all we can do is make sure this doesn't throw an exception
    from mpi4jax._src.flush import flush

    flush()
