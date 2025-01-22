Installation
============

Basic installation
------------------

Start by `installing a suitable version of JAX and jaxlib <https://github.com/google/jax#installation>`_. If you don't plan on using ``mpi4jax`` on GPU, the following will do:

.. code:: bash

   $ pip install 'jax[cpu]'

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

.. warning::

   If ``mpi4py`` is already installed, you *must* use ``--no-build-isolation`` when installing ``mpi4jax``:

   .. code:: bash

      # if mpi4py is already installed
      $ pip install cython
      $ pip install mpi4jax --no-build-isolation

To check which MPI library both libraries link to, run the following command in your
prompt.

.. code:: bash

	$ python -c "import mpi4py; print(mpi4py.get_config())"

If you wish to use a specific MPI library (only possible when using ``pip``), it is
usually sufficient to specify the ``MPICC`` environment variable *before* installing
``mpi4py``.

.. seealso::

   In doubt, please refer to `the mpi4py documentation <https://mpi4py.readthedocs.io/en/stable/install.html>`_.


Installation with NVIDIA GPU support (CUDA)
-------------------------------------------

.. note::

   There are 3 ways to install jax with CUDA support:
   - using a pypi-distributed CUDA installation (suggested by jax developers) ``pip install -U 'jax[cuda12]'`` 
   - using the locally-installed CUDA version, which must be compatible with jax. ``pip install -U 'jax[cuda12_local]'`` 
   The procedure to install ``mpi4jax`` for the two situations is different.

To use ``mpi4jax`` with pypi-distributed nvidia packages, which is the preferred way to install jax, you **must** install ``mpi4jax`` disabling
the build-time-isolation in order for it to link to the libraries in the nvidia-cuda-nvcc-cu12 package. To do so, run the following command:

.. code:: bash

   # assuming pip install -U 'jax[cuda12]' has been run
   $ pip install cython
   $ pip install mpi4jax --no-build-isolation

Alternatively, if you want to install ``mpi4jax`` with a locally-installed CUDA version, you can run the following command we need 
to be able to locate the CUDA headers on your system. If they are not detected automatically, you can set the environment 
variable :envvar:`CUDA_ROOT` when installing ``mpi4jax``::

   $ CUDA_ROOT=/usr/local/cuda pip install --no-build-isolation mpi4jax

This is sufficient for most situations. However, ``mpi4jax`` will copy all data from GPU to CPU and back before and after invoking MPI.

If this is a bottleneck in your application, you can build MPI with CUDA support and *communicate directly from GPU memory*. This requires that you re-build the entire stack:

- Your MPI library, e.g. `OpenMPI <https://www.open-mpi.org/faq/?category=buildcuda>`_, with CUDA support.
- ``mpi4py``, linked to your CUDA-enabled MPI installation.
- ``mpi4jax``, using the correct ``mpi4py`` installation.

.. seealso::

   Read :ref:`here <gpu-usage>` on how to use zero-copy GPU communication after installation.


Installation with Intel GPU/XPU support
---------------------------------------

``mpi4jax`` supports communication of JAX arrays stored in Intel GPU/XPU memory, via JAX's ``xpu`` backend.

**Requirements:**

- `Intel extension for OpenXLA <https://github.com/intel/intel-extension-for-openxla>`__ at least in version 0.3.0.
- SYCL headers and libraries, which come as part of the `Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html>`__.
- Optionally, `Intel MPI <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html>`__ with Intel XPU/GPU support.
  To leverage this, you also need to rebuild `mpi4py <https://mpi4py.readthedocs.io/en/stable/install.html>`__ to ensure it is linked to the XPU/GPU aware MPI implementation.

An example setup is found in the `mpi4jax test suite <https://github.com/mpi4jax/mpi4jax/tree/main/.github/workflows/build-xpu-ext.yml>`__.
