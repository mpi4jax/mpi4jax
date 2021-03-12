🔪 The Sharp Bits 🔪
====================

Read ahead for some pitfalls, counter-intuitive behavior, and sharp edges that we had to introduce in order to make this work.

.. _tokens:

Token management
----------------

The compiler behind JAX, XLA, is not aware of the fact that MPI function calls such as :func:`~mpi4jax.send` or :func:`~mpi4jax.recv` must be performed in a specific order (in jargon, that they have *side-effects*). It is therefore free to reorder those calls. Reordering of MPI calls usually leads to deadlocks, e.g. when both processes end up receiving before sending (instead of send-receive, receive-send).

*Tokens* are JAX's way to ensure that XLA does not re-order statements with side effects by injecting a fake data dependency between them.

This means that you *have* to use proper token management to prevent reordering from occurring. Every communication primitive in ``mpi4jax`` returns a token as the last return object, which you have to pass to subsequent primitives within the same JIT block, like this:

.. code:: python

    # DO NOT DO THIS
    mpi4jax.send(arr, comm=comm)
    new_arr, _ = mpi4jax.recv(arr, comm=comm)

    # INSTEAD, DO THIS
    token = mpi4jax.send(arr, comm=comm)
    new_arr, token = mpi4jax.recv(arr, comm=comm, token=token)

Those functions will then be executed in the same order as the sequence of tokens, from first to last.


No in-place operations in JAX
-----------------------------

JAX arrays are immutable, which means that functions cannot modify their input arguments. Therefore, unlike in ``mpi4py``, operations like :func:`mpi4jax.recv` use their first argument only to determine the correct shape and dtype of the output, but do not populate it with data.

This means that you *cannot* do:

.. code:: python

    # DO NOT DO THIS
    recv_arr = jnp.zeros((10, 10))
    mpi4jax.recv(recv_arr, comm=comm)
    # recv_arr still only contains 0

Instead, you need to use the returned array from :func:`mpi4jax.recv`:

.. code:: python

    # INSTEAD, DO THIS
    recv_arr = jnp.zeros((10, 10))
    recv_arr, _ = mpi4jax.recv(recv_arr, comm=comm)

.. _gpu-usage:

Using CUDA MPI
--------------

``mpi4jax`` is able to communicate data directly from and to GPU memory. :doc:`This requires that MPI, JAX, and mpi4jax are built with CUDA support. <installation>`

Currently, we cannot detect whether MPI was built with CUDA support.
Therefore, by default, ``mpi4jax`` will not read directly from GPU
memory, but instead copy to the CPU and back.

If you are certain that the underlying MPI library was built with CUDA
support, you can set the following environment variable:

.. code:: bash

   $ export MPI4JAX_USE_CUDA_MPI=1

Data will then be copied directly from GPU to GPU. If your MPI library
does not have CUDA support, you will receive a segmentation fault when
trying to access GPU memory.


Using ``mpi4jax`` *and* ``mpi4py``
----------------------------------

.. warning::

    Do not use ``mpi4jax`` and ``mpi4py`` with the same communicator!

Consider the following example, where one process sends some Python data via ``mpi4py`` and JAX data via ``mpi4jax``, and the other process receives it:

.. code:: python

    # DO NOT DO THIS
    import numpy as np
    import jax.numpy as jnp

    from mpi4py import MPI
    import mpi4jax

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    arr_np = np.random.rand(10, 10)
    arr_jax = jnp.zeros((10, 10))

    if rank == 0:
        mpi4jax.send(arr_jax, comm=comm)
        comm.send(arr_np)
    else:
        arr_jax = mpi4jax.recv(arr_jax, comm=comm)
        arr = comm.recv(arr_np)

Because everything is lazily executed in JAX, we cannot rely on a particular execution order. Specifically, we don't know whether the function ``mpi4jax.send`` wille be executed before or after the ``comm.send`` call. In the worst case, this creates a deadlock.

The simplest solution is therefore to stick to *either* ``mpi4py`` *or* ``mpi4jax``. But if you have to use both, make sure that they use different communicators:


.. code:: python

    # INSTEAD, DO THIS
    import numpy as np
    import jax.numpy as jnp

    from mpi4py import MPI
    import mpi4jax

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # create a new communicator for mpi4jax
    comm_jax = comm.Clone()

    arr_np = np.random.rand(10, 10)
    arr_jax = jnp.zeros((10, 10))

    if rank == 0:
        mpi4jax.send(arr_jax, comm=comm_jax)
        comm.send(arr_np)
    else:
        arr_jax = mpi4jax.recv(arr_jax, comm=comm_jax)
        arr = comm.recv(arr_np)

    comm_jax.Free()
