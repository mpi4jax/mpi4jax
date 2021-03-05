mpi4jax
=======

|Tests| |codecov| |Conda Recipe| |Documentation Status|

MPI plugin for JAX, allowing MPI operations to be inserted in blocks
compiled with :func:`jax.jit`.

Installation
------------

``mpi4jax`` is available through ``pip`` and ``conda``:

.. code:: bash

   $ pip install mpi4jax                     # Pip
   $ conda install -c conda-forge mpi4jax    # conda

Refer to the documentation for more advanced installation examples.


Example usage
-------------

.. code:: python

   from mpi4py import MPI
   import jax
   import mpi4jax

   comm = MPI.COMM_WORLD
   a = jax.numpy.ones(5,4)
   b, token = mpi4jax.Allreduce(a, op=MPI.SUM, comm=comm)
   b_jit, token = jax.jit(lambda x: mpi4jax.Allreduce(x, op=MPI.SUM, comm=comm))(a)


`See all supported operations here. <https://mpi4jax.readthedcs.org/en/latest/api.html>`

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
