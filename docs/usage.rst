Usage
=====

A basic example
---------------

Token management
----------------

Tokens are JAX's way to ensure that XLA (the underlying compiler) does not re-order statements with side effects. Re-ordering MPI calls usually leads to deadlocks, e.g. when both processes end up receiving before sending (instead of send-receive, receive-send).

This means that you *have* to use proper token management to prevent this from happening. Every communication primitive in mpi4jax returns a token as the last return argument, which you have to pass to subsequent primitives within the same JIT block.

.. code:: python

    # DO NOT DO THIS
    mpi4jax.Send(arr, comm=comm)
    new_arr, _ = mpi4jax.Recv(arr, comm=comm)

    # INSTEAD, DO THIS
    token = mpi4jax.Send(arr, comm=comm)
    new_arr, token = mpi4jax.Recv(arr, comm=comm, token=token)
