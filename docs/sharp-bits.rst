🔪 The Sharp Bits 🔪
====================

What are tokens?
----------------

Tokens are JAX's way to ensure that XLA (the underlying compiler) does not re-order statements with side effects. Re-ordering MPI calls usually leads to deadlocks, e.g. when both processes end up receiving before sending (instead of send-receive, receive-send).

This means that you *have* to use proper token management to prevent this from happening:

.. code::python

    # DO NOT DO THIS
    mpi4jax.Send(arr, comm=comm)
    new_arr, _ = mpi4jax.Recv(arr, comm=comm)

    # INSTEAD, DO THIS
    token = mpi4jax.Send(arr, comm=comm)
    new_arr, token = mpi4jax.Recv(arr, comm=comm, token=token)



Communicating GPU arrays
------------------------


Using mpi4jax *and* mpi4py
--------------------------

.. warning::

    Do not use mpi4jax and mpi4py with the same communicator.

Consider the following example:

.. code:: python

    import numpy as np
    from mpi4py import MPI
    import mpi4jax

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arr = np.random.rand(10, 10)

    if rank == 0:
        mpi4jax.Send(arr, comm=comm)
        arr = comm.Recv(arr)
    else:
        arr = comm.Recv(arr)
        mpi4jax.Send(arr, comm=comm)

Because everything is lazily executed in JAX, you cannot rely on a particular execution order. Specifically, you don't know whether the function ``mpi4jax.Send`` wille be executed before or after the ``comm.Recv`` call. In the worst case, this creates a deadlock.

The simplest solution is of course to stick to *either* mpi4py *or* mpi4jax. But if you have to use both, make sure that they use different communicators.


I don't want to use omnistaging, but mpi4jax says I have to :(
--------------------------------------------------------------
