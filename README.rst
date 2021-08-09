mpi4jax
=======

|PyPI Version| |Conda Version| |Tests| |codecov| |Documentation Status|

``mpi4jax`` enables zero-copy, multi-host communication of `JAX <https://jax.readthedocs.io/>`_ arrays, even from jitted code and from GPU memory.


But why?
--------

The JAX framework `has great performance for scientific computing workloads <https://github.com/dionhaefner/pyhpc-benchmarks>`_, but its `multi-host capabilities <https://jax.readthedocs.io/en/latest/jax.html#jax.pmap>`_ are still limited.

With ``mpi4jax``, you can scale your JAX-based simulations to *entire CPU and GPU clusters* (without ever leaving ``jax.jit``).

In the spirit of differentiable programming, ``mpi4jax`` also supports differentiating through some MPI operations.


Installation
------------

``mpi4jax`` is available through ``pip`` and ``conda``:

.. code:: bash

   $ pip install mpi4jax                     # Pip
   $ conda install -c conda-forge mpi4jax    # conda

If you use pip and don't have JAX installed already, you will also need to do:

.. code:: bash

   $ pip install jaxlib

(or an equivalent GPU-enabled version, `see the JAX installation instructions <https://github.com/google/jax#installation>`_)

In case your MPI installation is not detected correctly, `it can help to install mpi4py separately <https://mpi4py.readthedocs.io/en/stable/install.html>`_. When using a pre-installed ``mpi4py``, you *must* use ``--no-build-isolation`` when installing ``mpi4jax``:

.. code:: bash

   # if mpi4py is already installed
   $ pip install cython
   $ pip install mpi4jax --no-build-isolation

`Our documentation includes some more advanced installation examples. <https://mpi4jax.readthedocs.io/en/latest/installation.html>`_


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

``allreduce`` is just one example of the MPI primitives you can use. `See all supported operations here <https://mpi4jax.readthedocs.org/en/latest/api.html>`_.

Community guidelines
--------------------

If you have a question or feature request, or want to report a bug, feel free to `open an issue <https://github.com/mpi4jax/mpi4jax/issues>`_.

We welcome contributions of any kind `through pull requests <https://github.com/mpi4jax/mpi4jax/pulls>`_. For information on running our tests, debugging, and contribution guidelines please `refer to the corresponding documentation page <https://mpi4jax.readthedocs.org/en/latest/developers.html>`_.


.. |Tests| image:: https://github.com/mpi4jax/mpi4jax/workflows/Tests/badge.svg
   :target: https://github.com/mpi4jax/mpi4jax/actions?query=branch%3Amaster
.. |codecov| image:: https://codecov.io/gh/mpi4jax/mpi4jax/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mpi4jax/mpi4jax
.. |PyPI Version| image:: https://img.shields.io/pypi/v/mpi4jax
   :target: https://pypi.org/project/mpi4jax/
.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/mpi4jax.svg
   :target: https://anaconda.org/conda-forge/mpi4jax
.. |Documentation Status| image:: https://readthedocs.org/projects/mpi4jax/badge/?version=latest
   :target: https://mpi4jax.readthedocs.io/en/latest/?badge=latest
