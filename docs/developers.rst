Developer guide
===============

Development install
-------------------

To install mpi4jax in editable mode along with all optional dependencies for testing, just run

.. code:: bash

   $ pip install -e .[dev]

from the repository root.

Running tests
-------------

We use ``pytest`` for testing. After installing the development dependencies, you can run our testing suite with the following commands:

.. code:: bash

    $ pytest .
    $ mpirun -np 2 pytest .

.. warning::

    Just executing ``pytest`` will run the tests on only 1 process, which means that a large part of mpi4jax cannot be tested (because it relies on communication between different processes). Therefore, you should always make sure that the tests also pass on multiple processes (via ``mpirun``).

Contributing
------------

We welcome code contributions or changes to the documentation via `pull requests <https://github.com/mpi4jax/mpi4jax/pulls>`_ (PRs).

We use pre-commit hooks to enforce a common code format. To install
them, just run:

.. code:: bash

   $ pre-commit install

in the repository root. Then, all changes will be validated automatically before you commit.

.. note::

    If you introduce new code, please make sure that it is covered by tests.
    To catch problems early, we recommend that you run the test suite locally before creating a PR.

Debugging
---------

You can set the environment variable ``MPI4JAX_DEBUG`` to ``1`` to
enable debug logging every time an MPI primitive is called from within a
jitted function. You will then see messages like these:

.. code:: bash

   $ MPI4JAX_DEBUG=1 mpirun -n 2 python send_recv.py
   r0 | MPI_Send -> 1 with tag 0 and token 7fd7abc5f5c0
   r1 | MPI_Recv <- 0 with tag -1 and token 7f9af7419ac0

This can be useful to debug deadlocks or MPI errors.
