---
title: 'mpi4jax: Zero-copy communication of JAX arrays'
tags:
  - Python
  - JAX
  - MPI
  - high performance computing
  - parallel computing
authors:
  - name: Filippo Vicentini^[Contributed equally, order determined by coin flip.]
    affiliation: 1
  - name: Dion Häfner^[Contributed equally, order determined by coin flip.]
    orcid: 0000-0002-4465-7317
    affiliation: 2
affiliations:
 - name: Affiliation 1
   index: 1
 - name: Niels Bohr Institute, Copenhagen University, Copenhagen, Denmark
   index: 2
date: 16 March 2021
bibliography: paper.bib

---

# Summary

The tensor framework JAX shows excellent performance on both machine learning and scientific computing workloads, while all user code is written in pure Python.

However, machine learning and high-performance computing are still being run on very different hardware stacks. While machine learning is typically done on few highly parallel units (GPUs or TPUs), high-performance workloads such as physical models tend to run on clusters of dozens to thousands of CPUs. Unfortunately, support from JAX and the underlying compiler XLA is much more mature in the former case. Notably, there is no built-in solution to communicate data between different nodes that is as sophisticated as the widely used MPI (message passing interface) libraries.

Here, we present `mpi4jax` to fill this gap. `mpi4jax` uses XLA's custom call mechanism to register the most important MPI primitives as JAX primitives. This means that users can communicate arbitrary JAX data without performance and usability penalty. In particular, `mpi4jax` is able to communicate without copying from CPU and GPU memory (if built against a CUDA-aware MPI library) between one or multiple hosts (e.g. via an Infiniband network on a cluster).

This also means that existing applications using e.g. NumPy and `mpi4py` can be ported seamlessly to the JAX ecosystem for potentially significant performance gains.

# Statement of Need

For decades, high-performance computing has been done in low-level programming languages like Fortran or C. But the ubiquity of Python is starting to spill into this domain as well, and for good reason, being the de-facto programming lingua franca of science. With a combination of NumPy and `mpi4py`, Python users can build massively parallel applications without delving into low-level programming languages, which is often advantageous when human time is more valuable than computer time. But it is of course unsatisfying to leave possible performance on the table.

Google's JAX library leverages the XLA compiler and supports just-in-time compilation (JIT) of Python code to XLA primitives. [The result is highly competitive performance on both CPU and GPU.](https://github.com/dionhaefner/pyhpc-benchmarks) This often achieves the dream of high-performance computing --- low-level performance in high-level code.

Two real-world use cases for `mpi4jax` are the ocean model Veros [@hafner_veros_2018] and the many-body quantum systems toolkit netket [@carleo_netket_2019]:

- In the case of Veros, MPI primitives are needed to communicate overlapping grid cells between processes. Communication primitives are buried deep into the physical subroutines. Therefore, refactoring the codebase to leave `jax.jit` every time data needs to be communicated would severely break the control flow of the model and presumably incur a hefty performance loss (in addition to the cost of copying data from and to JAX). Through `mpi4jax`, it is possible to apply the JIT compiler to whole subroutines to avoid this entirely.

- netket...  **write me**


# Implementation

In essence, `mpi4jax` combines JAX's custom call mechanism with `mpi4py.libmpi` (which exposes MPI C primitives as Cython callables).

The implementation of a primitive in `mpi4jax` consists of two parts:

1. A Python module that registers a new primitive with JAX. JAX primitives consist of several parts, such as an abstract evaluation rule (used to infer output shapes and data types), and 2 translation rules (one for each CPU and GPU) that convert inputs to the appropriate XLA-compatible types. In particular, we need to ensure that all numerical data types are of the correct, expected type (e.g., casting Python integers to the equivalent of the C type `uintptr_t`). Optionally, we can also define transpose and differentiation rules (if applicable).

2. A Cython function that casts raw input arguments passed by XLA to their true C type, so they can be passed on to MPI. On CPU, arguments are given in the form of arrays of void pointers, `void**`, so we use static casts for conversion. On GPU, input data is given as a raw char array, `char*`, which we deserialize to a custom Cython `struct` whose fields represent the input data.

  On GPU, our Cython bridge also supports copying the data from device to host and back before and after calling MPI (by linking `mpi4jax` to the CUDA runtime library). This way, we support the communication of GPU data via main memory if the underlying MPI library is not built with CUDA support (at a small performance penalty).

This yields MPI primitives that are callable from compiled JAX code. However, there is one additional complication: we need to take special care to ensure that MPI statements are not re-ordered. Consider the following example:

```python
@jax.jit
def exchange_data(arr):
   if rank == 0:
      mpi4jax.send(arr, dest=1)
      newarr = mpi4jax.recv(arr, source=1)
   else:
      newarr = mpi4jax.recv(arr, source=0)
      mpi4jax.send(arr, dest=0)
   return newarr
```

As JAX and XLA operate on the assumption that all primitives are pure functions without side effects, the compiler is in principle free to re-order the `send` and `recv` statements above. This would typically lead to a deadlock or crash, as e.g. both processes might wait for each others' input indefinitely.

The solution to this in JAX is a token mechanism that involves threading a dummy token value through each primitive. This introduces a fake data dependency between subsequent calls using the token, which prevents XLA from re-ordering them relative to each other.

The example above, using proper token management, reads:

```python
@jax.jit
def exchange_data(arr):
   if rank == 0:
      token = mpi4jax.send(arr, dest=1)
      newarr, token = mpi4jax.recv(arr, source=1, token=token)
   else:
      newarr, token = mpi4jax.recv(arr, source=0)
      token = mpi4jax.send(arr, dest=0, token=token)
   return newarr
```

As a result, we are successfully able to execute MPI primitives just as if they were JAX primitives.

As of yet, we support the MPI operations `allgather`, `allreduce`, `alltoall`, `bcast`, `gather`, `recv`, `reduce`, `scan`, `scatter`, `send`, and `sendrecv`. Most currently unsupported operations such as `gatherv` could be implemented with little additional work if needed by an application (since all fundamental obstacles are already solved).

# Example & Benchmark: Non-linear Shallow Water Solver

As a prototype, and to use as a benchmark, we have ported a non-linear shallow water solver to JAX and parallelized it with `mpi4jax` (\autoref{fig:shallow-water}).

![Output snapshot of the non-linear shallow water model. Shading indicates surface height, quivers show the current's velocity field. \label{fig:shallow-water}](shallow-water.pdf){ width=80% }

The full example is available in the `mpi4jax` documentation. It defines a function `enforce_boundaries` which handles halo exchanges between all MPI processes. The core of it reads something like this (plus some additional code to handle processes with at the edges of the domain):

```python
@jax.jit
def enforce_boundaries(arr, grid, token):
   for send_proc, recv_proc in proc_neighbors:
      recv_idx = overlap_slices_recv[recv_dir]
      recv_arr = jnp.empty_like(arr[recv_idx])

      send_idx = overlap_slices_send[send_dir]
      send_arr = arr[send_idx]

      recv_arr, token = mpi4jax.sendrecv(
          send_arr,
          recv_arr,
          source=recv_proc,
          dest=send_proc,
          comm=mpi_comm,
          token=token,
      )
      # update solution
      arr = arr.at[recv_idx].set(recv_arr)
  return arr
```

Then, it can be used in the physical simulation like this:

```python
@partial(jax.jit, static_argnums=(1,))
def shallow_water_step(state, is_first_step):
   token = jax.lax.create_token()
   # ...
   fe = fe.at[1:-1, 1:-1].set(0.5 * (hc[1:-1, 1:-1] + hc[1:-1, 2:]) * u[1:-1, 1:-1])
   fn = fn.at[1:-1, 1:-1].set(0.5 * (hc[1:-1, 1:-1] + hc[2:, 1:-1]) * v[1:-1, 1:-1])
   fe, token = enforce_boundaries(fe, "u", token)
   fn, token = enforce_boundaries(fn, "v", token)
   # ...
```



| Platform | # processes | Time (s) | Rel. speedup |
|----------|-------------|---------:|-------------:|
| CPU      | 1 (NumPy)   | 770      | 1            |
|          |             |          |              |
| CPU      | 1           | 112      | 6.9          |
|          | 2           | 90       | 8.6          |
|          | 4           | 39       | 19           |
|          | 6           | 29       | 27           |
|          | 8           | 21       | 37           |
|          | 16          | 16       | 48           |
|          |             |          |              |
| GPU      | 1           | 6.3      | 122          |
|          | 2           | 3.9      | 197          |



# Outlook

In the previous sections, we introduced `mpi4jax`, which allows zero-copy communication of JAX-owned data. `mpi4jax` provides an implementation of the most important MPI operations in a way that is usable from JAX compiled code.

However, JAX is more than just a JIT compiler.

# Acknowledgements

We thank all JAX developers in general and Matthew Johnson and Peter Hawkins in particular for their outstanding support on the many issues we opened.

DH received support from the Danish Hydrocarbon Research and Technology Centre (DHRTC).

# References
