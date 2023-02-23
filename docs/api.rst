API Reference
=============

Utilities
---------

has_cuda_support
++++++++++++++++

.. autofunction:: mpi4jax.has_cuda_support


Communication primitives
------------------------

allgather
+++++++++

.. autofunction:: mpi4jax.allgather

allreduce
+++++++++

.. autofunction:: mpi4jax.allreduce

alltoall
++++++++

.. autofunction:: mpi4jax.alltoall

barrier
+++++++

.. autofunction:: mpi4jax.barrier

bcast
+++++

.. autofunction:: mpi4jax.bcast

gather
++++++

.. autofunction:: mpi4jax.gather

recv
++++

.. autofunction:: mpi4jax.recv

reduce
++++++

.. autofunction:: mpi4jax.reduce

scan
++++

.. autofunction:: mpi4jax.scan

scatter
+++++++

.. autofunction:: mpi4jax.scatter

send
++++

.. autofunction:: mpi4jax.send

sendrecv
++++++++

.. autofunction:: mpi4jax.sendrecv


Experimental
------------

auto_tokenize
+++++++++++++

.. warning::

    ``auto_tokenize`` is currently broken for JAX 0.4.4 and later.
    To use it, downgrade to ``jax<=0.4.3``.
    See `issue #192 <https://github.com/mpi4jax/mpi4jax/issues/192>` for more details.

.. autofunction:: mpi4jax.experimental.auto_tokenize
