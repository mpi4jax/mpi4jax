mpi4jax
=======

|JOSS paper| |PyPI Version| |Conda Version| |Tests| |codecov| |Documentation Status|

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

Depending on the different jax backends you want to use, you can install mpi4jax in the following way

.. code:: bash

   # pip install 'jax[cpu]'
   $ pip install mpi4jax

   # pip install -U 'jax[cuda12]'
   $ pip install cython
   $ pip install mpi4jax --no-build-isolation

   # pip install -U 'jax[cuda12_local]'
   $ CUDA_ROOT=XXX pip install mpi4jax

(for more informations on jax GPU distributions, `see the JAX installation instructions <https://github.com/google/jax#installation>`_)

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

How to cite
-----------

If you use ``mpi4jax`` in your work, please consider citing the following article:

::

  @article{mpi4jax,
    doi = {10.21105/joss.03419},
    url = {https://doi.org/10.21105/joss.03419},
    year = {2021},
    publisher = {The Open Journal},
    volume = {6},
    number = {65},
    pages = {3419},
    author = {Dion Häfner and Filippo Vicentini},
    title = {mpi4jax: Zero-copy MPI communication of JAX arrays},
    journal = {Journal of Open Source Software}
  }

.. |Tests| image:: https://github.com/mpi4jax/mpi4jax/workflows/Tests/badge.svg
   :target: https://github.com/mpi4jax/mpi4jax/actions?query=branch%3Amain
.. |codecov| image:: https://codecov.io/gh/mpi4jax/mpi4jax/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/mpi4jax/mpi4jax
.. |PyPI Version| image:: https://img.shields.io/pypi/v/mpi4jax
   :target: https://pypi.org/project/mpi4jax/
.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/mpi4jax.svg
   :target: https://anaconda.org/conda-forge/mpi4jax
.. |Documentation Status| image:: https://readthedocs.org/projects/mpi4jax/badge/?version=latest
   :target: https://mpi4jax.readthedocs.io/en/latest/?badge=latest
.. |JOSS paper| image:: https://joss.theoj.org/papers/10.21105/joss.03419/status.svg
   :target: https://doi.org/10.21105/joss.03419
