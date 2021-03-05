🔪 The Sharp Bits 🔪
====================

Zero-copy GPU communication
---------------------------

mpi4jax is able to communicate data directly from and to GPU memory. This requires that MPI, JAX, and mpi4jax are built with CUDA support.

``mpi4jax`` also supports JAX arrays stored in GPU memory. To use JAX on
the GPU, make sure that your ``jaxlib`` is `built with CUDA
support <https://github.com/google/jax#pip-installation>`__.

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


Using mpi4jax *and* mpi4py
--------------------------

.. warning::

    In short: Do not use ``mpi4jax`` and ``mpi4py`` with the same communicator!

Consider the following example, where one process sends some data and then receives, and the other receives and then sends:

.. code:: python

    import numpy as np
    import jax.numpy as jnp

    from mpi4py import MPI
    import mpi4jax

    comm = MPI.COMM_WORLD
    comm_jax = comm.Clone()
    rank = comm.Get_rank()

    arr_np = np.random.rand(10, 10)
    arr_jax = jnp.zeros((10, 10))

    if rank == 0:
        mpi4jax.Send(arr_jax, comm=comm_jax)
        comm.Send(arr_np)
    else:
        arr_jax = mpi4jax.Recv(arr_jax, comm=comm_jax)
        arr = comm.Recv(arr_np)

    comm_jax.Free()

Because everything is lazily executed in JAX, we cannot rely on a particular execution order. Specifically, we don't know whether the function ``mpi4jax.Send`` wille be executed before or after the ``comm.Recv`` call. In the worst case, this creates a deadlock.

The simplest solution is therefore to stick to *either* ``mpi4py`` *or* ``mpi4jax``. But if you have to use both, make sure that they use different communicators:


.. code:: python

    import numpy as np
    import jax.numpy as jnp

    from mpi4py import MPI
    import mpi4jax

    comm = MPI.COMM_WORLD
    comm_jax = comm.Clone()
    rank = comm.Get_rank()

    arr_np = np.random.rand(10, 10)
    arr_jax = jnp.zeros((10, 10))

    if rank == 0:
        mpi4jax.Send(arr_jax, comm=comm_jax)
        comm.Send(arr_np)
    else:
        arr_jax = mpi4jax.Recv(arr_jax, comm=comm_jax)
        arr = comm.Recv(arr_np)

    comm_jax.Free()
