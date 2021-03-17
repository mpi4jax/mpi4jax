Installation
============

Basic installation
------------------

Start by `installing a suitable version of JAX and jaxlib <https://github.com/google/jax#installation>`_. If you don't plan on using ``mpi4jax`` on GPU, the following will do:

.. code:: bash

   $ pip install jax jaxlib

.. note::

   Much of the functionality we need has recently been added to JAX, which itself is changing frequently. Therefore, ``mpi4jax`` has somewhat strict requirements on the supported versions of JAX and jaxlib. Be prepared to upgrade!

We recommend that you use ``pip`` to install mpi4jax (but a distribution is also available via ``conda`` which will work if MPI, mpi4py and mpi4jax are all installed through conda ):

.. code:: bash

   $ pip install mpi4jax
   $ conda install -c conda-forge mpi4jax

Installing via ``pip`` requires a working installation of MPI to succeed. If you don't already have MPI and want to get started as quickly as possible, try ``conda``, which bundles the MPI library (but remember `not to mix pip and conda <https://www.anaconda.com/blog/using-pip-in-a-conda-environment>`_).

.. warning::

   We advise against using the conda installation in HPC environments because it is not possible to change the MPI library ``mpi4py`` is linked against.

And that is it! If you are familiar with MPI, you should in principle be able to get started right away. However, we recommend that you have a look at :doc:`sharp-bits`, to make sure that you are aware of some of the pitfalls of ``mpi4jax``.

Selecting the MPI distribution
------------------------------

``mpi4jax`` will use the MPI distribution with which ``mpi4py`` was built.
If ``mpi4py`` is not installed, it will be installed automatically before
installing ``mpi4jax``.

To check which MPI library both libraries link to, run the following command in your
prompt.

.. code:: bash

	$ python -c "import mpi4py; print(mpi4py.get_config())"

If you wish to use a specific MPI library (only possible when using ``pip``), it is
usually sufficient to specify the ``MPICC`` environment variable `before` installing
```mpi4py``.

.. seealso::

   In doubt, please refer to `the mpi4py documentation <https://mpi4py.readthedocs.io/en/stable/install.html>`_.


Installation with GPU support
-----------------------------

.. note::

   To use JAX on the GPU, make sure that your ``jaxlib`` is `built with CUDA support <https://github.com/google/jax#installation>`_.

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
