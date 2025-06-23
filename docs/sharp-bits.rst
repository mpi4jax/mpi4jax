🔪 The Sharp Bits 🔪
====================

Read ahead for some pitfalls, counter-intuitive behavior, and sharp edges that we had to introduce in order to make this work.

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

Using Intel XPU aware MPI
~~~~~~~~~~~~~~~~~~~~~~~~~

``mpi4jax`` is able to communicate data directly from and to Intel XPU
and Intel GPU memory. This requires that you have installed MPI that is
Intel GPU/XPU aware (MPI calls can work directly with XPU/GPU memory)
and that JAX and `mpi4jax is built with Intel XPU
support <installation>`__.

Currently, we cannot detect whether MPI is XPU/GPU aware. Therefore, by
default, ``mpi4jax`` will not read directly from XPU/GPU memory, but
instead copy to the CPU and back.

If you are certain that the underlying MPI library is XPU/GPU aware
then, you can set the following environment variable:

.. code:: bash

   $ export MPI4JAX_USE_SYCL_MPI=1

Data will then be copied directly from XPU to XPU. If your MPI library
cannot work with Intel GPU/XPU buffers, you will receive a segmentation
fault when trying to access mentioned GPU/XPU memory.

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
