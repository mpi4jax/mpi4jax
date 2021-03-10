mpi4jax
=======

|Tests| |codecov| |Conda Recipe| |Documentation Status|

``mpi4jax`` enables zero-copy, multi-host communication of `JAX <https://jax.readthedocs.io/>`_ arrays, even from jitted code and from GPU memory.


But why?
--------

The JAX framework `has great performance for scientific computing workloads <https://github.com/dionhaefner/pyhpc-benchmarks>`_, but its `multi-host capabilities <https://jax.readthedocs.io/en/latest/jax.html#jax.pmap>`_ are still limited.

With ``mpi4jax``, you can scale your JAX-based simulations to *entire CPU and GPU clusters* (without ever leaving ``jax.jit``).

In the spirit of differentiable programming, ``mpi4jax`` also supports differentiating through some MPI operations.


Quick installation
------------------

``mpi4jax`` is available through ``pip`` and ``conda``:

.. code:: bash

   $ pip install mpi4jax                     # Pip
   $ conda install -c conda-forge mpi4jax    # conda

Our documentation includes some more advanced installation examples.


Example usage
-------------

.. code:: python

   from mpi4py import MPI
   import jax
   import jax.numpy as jnp
   import mpi4jax

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()

   @jax.jit
   def foo(arr):
      arr = arr + rank
      arr_sum, _ = mpi4jax.allreduce(arr, op=MPI.SUM, comm=comm)
      return arr_sum

   a = jnp.zeros((3, 3))
   result = foo(a)

   if rank == 0:
      print(result)

Running this script on 4 processes gives:

.. code:: bash

   $ mpirun -n 4 python example.py
   [[6. 6. 6.]
    [6. 6. 6.]
    [6. 6. 6.]]

``allreduce`` is just one example of the MPI primitives you can use. `See all supported operations here. <https://mpi4jax.readthedocs.org/en/latest/api.html>`_


Contributing
------------

We use pre-commit hooks to enforce a common code format. To install
them, just run:

.. code:: bash

   $ pip install pre-commit
   $ pre-commit install


Debugging
---------

You can set the environment variable ``MPI4JAX_DEBUG`` to ``1`` to
enable debug logging every time an MPI primitive is called from within a
jitted function. You will then see messages like this:

.. code:: bash

   $ MPI4JAX_DEBUG=1 mpirun -n 2 python send_recv.py
   r0 | MPI_Send -> 1 with tag 0 and token 7fd7abc5f5c0
   r1 | MPI_Recv <- 0 with tag -1 and token 7f9af7419ac0


Contributors
------------

-  Filippo Vicentini `@PhilipVinc <https://github.com/PhilipVinc>`_
-  Dion Häfner `@dionhaefner <https://github.com/dionhaefner>`_

.. |Tests| image:: https://github.com/PhilipVinc/mpi4jax/workflows/Tests/badge.svg
.. |codecov| image:: https://codecov.io/gh/PhilipVinc/mpi4jax/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/PhilipVinc/mpi4jax
.. |Conda Recipe| image:: https://img.shields.io/badge/recipe-mpi4jax-green.svg
   :target: https://anaconda.org/conda-forge/mpi4jax
.. |Documentation Status| image:: https://readthedocs.org/projects/mpi4jax/badge/?version=latest
   :target: https://mpi4jax.readthedocs.io/en/latest/?badge=latest
