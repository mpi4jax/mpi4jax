Installation
============

Basic installation
------------------

.. warning::

   Much of the functionality we need has recently been added to JAX, which itself is changing frequently. Therefore, ``mpi4jax`` has somewhat strict requirements on the supported versions of JAX and jaxlib.

We recommend that you use ``pip`` or ``conda`` to install mpi4jax:

.. code:: bash

   $ pip install mpi4jax
   $ conda install -c conda-forge mpi4jax

Installing via ``pip`` usually requires a working installation of MPI to succeed. If you don't already have MPI and want to get started as quickly as possible, try ``conda``, which bundles the MPI library.

And that is it! If you are familiar with MPI, you should in principle be able to get started right away. However, we recommend that you have a look at :doc:`sharp-bits`, to make sure that you are aware of some of the pitfalls of ``mpi4jax``.

Installation with GPU support
-----------------------------

.. note::

   To use JAX on the GPU, make sure that your ``jaxlib`` is `built with CUDA support <https://github.com/google/jax#pip-installation>`_.

``mpi4jax`` also supports JAX arrays stored in GPU memory.

To build ``mpi4jax``'s GPU extensions, we need to be able to locate the CUDA headers on your system. If they are not detected automatically, you can set the environment variable :envvar:`CUDA_ROOT` when installing ``mpi4jax``::

   $ CUDA_ROOT=/usr/local/cuda pip install mpi4jax

This is sufficient for most situations. However, ``mpi4jax`` will copy all data from GPU to CPU and back before and after invoking MPI.

If this is a bottleneck in your application, you can build MPI with CUDA support and *communicate directly from GPU memory*. This requires that you re-build the entire stack:

- Your MPI library, e.g. `OpenMPI <https://www.open-mpi.org/faq/?category=buildcuda>`_, with CUDA support.
- ``mpi4py``, linked to your CUDA-enabled MPI installation.
- ``mpi4jax``, using the correct ``mpi4py`` installation.

.. seealso::

   Read :ref:`here <gpu-usage>` on how to use zero-copy GPU communication after installation.
