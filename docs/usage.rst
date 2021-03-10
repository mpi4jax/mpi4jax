Usage examples
==============

Basic example: Global sum
-------------------------

The following computes the sum of an array over several processes (similar to :func:`jax.lax.psum`), using :func:`~mpi4jax.allreduce`:

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

Most MPI libraries supply a wrapper executable ``mpirun`` to execute a script on several processes:

.. code:: bash

   $ mpirun -n 4 python mpi4jax-example.py
   [[6. 6. 6.]
    [6. 6. 6.]
    [6. 6. 6.]]

The result is an array full of the value 6, because each process adds its rank to the result (4 processes with ranks 0, 1, 2, 3).

Basic example: sending and receiving
------------------------------------

``mpi4jax`` can of course also send and receive data without performing an operation on it. For this, you can use :func:`~mpi4jax.send` and :func:`~mpi4jax.recv`:

.. _example_2:

.. code:: python

    from mpi4py import MPI
    import jax
    import jax.numpy as jnp
    import mpi4jax

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    assert size == 2  # make sure we are on 2 processes

    @jax.jit
    def foo(arr):
        arr = arr + rank
        # note: this could also use mpi4jax.sendrecv
        if rank == 0:
            # send, then receive
            token = mpi4jax.send(arr, dest=1, comm=comm)
            other_arr, token = mpi4jax.recv(arr, source=1, comm=comm, token=token)
        else:
            # receive, then send
            other_arr, token = mpi4jax.recv(arr, source=0, comm=comm)
            token = mpi4jax.send(arr, dest=0, comm=comm, token=token)

        return other_arr

    a = jnp.zeros((3, 3))
    result = foo(a)

    print(f'r{rank} | {result}')

Executing this shows that each process has received the data from the other process:

.. code:: bash

    $ mpirun -n 2 python mpi4jax-example-2.py
    r1 | [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    r0 | [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]

For operations like this, the correct order of the :func:`~mpi4jax.send` / :func:`~mpi4jax.recv` calls is critical to prevent the program from deadlocking (e.g. when both processes wait for data at the same time).

In ``mpi4jax``, we enforce order of execution through *tokens*. In :ref:`the example code <example_2>`, you can see this behavior e.g. in the following lines:

.. code:: python

    token = mpi4jax.send(arr, dest=1, comm=comm)
    other_arr, token = mpi4jax.recv(arr, source=1, comm=comm, token=token)

The first call to :func:`~mpi4jax.send` returns a token, which we then pass to :func:`~mpi4jax.recv`. :func:`~mpi4jax.recv` *also* returns a new token that we could pass to subsequent communication primitives.

Because of the nature of JAX, **using tokens to enforce order is not optional.** If you do not use correct token management, you will experience deadlocks and crashes.

.. seealso::

    For more information on tokens, see :ref:`tokens`.
