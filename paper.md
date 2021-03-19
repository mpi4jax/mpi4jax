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

The tensor math accelerator framework JAX shows excellent performance on both machine learning and scientific computing workloads, while all user code is written in pure Python.

However, machine learning and high-performance computing are still being run on very different hardware stacks. While machine learning is typically done on few highly parallel units (GPUs or TPUs), high-performance workloads such as physical models tend to run on clusters of dozens to thousands of CPUs. Unfortunately, support from JAX and the underlying compiler XLA is much more mature in the former case. Notably, there is no built-in solution to communicate data between different nodes that is as sophisticated as the widely used MPI (message passing interface) libraries.

Here, we present `mpi4jax` to fill this gap. `mpi4jax` uses XLA's custom call mechanism to register the most important MPI primitives as JAX primitives. This means that users can communicate arbitrary JAX data without performance and usability penalty. In particular, `mpi4jax` is able to communicate without copying from CPU and GPU memory (if built against a CUDA-aware MPI library) between one or multiple hosts (e.g. via an Infiniband network on a cluster).

This also means that existing applications using e.g. NumPy and `mpi4py` can be ported seamlessly to the JAX ecosystem for potentially significant performance gains.

# Statement of Need

For decades, high-performance computing has been done in low-level programming languages like Fortran or C. But the ubiquity of Python is starting to spill into this domain as well, and for good reason, being the de-facto programming lingua franca of science. With a combination of NumPy and `mpi4py`, Python users can build massively parallel applications without delving into low-level programming languages, which is often advantageous when human time is more valuable than computer time. But it is of course unsatisfying to leave possible performance on the table.

Google's JAX library leverages the XLA compiler and supports just-in-time compilation (JIT) of Python code to XLA primitives. [The result is highly competitive performance on both CPU and GPU.](https://github.com/dionhaefner/pyhpc-benchmarks) This often achieves the dream of high-performance computing --- low-level performance in high-level code.

Two real-world use cases for `mpi4jax` are the ocean model Veros [@hafner:2018] and the many-body quantum systems toolkit netket [@carleo:2019]:

- In the case of Veros, MPI primitives are needed to communicate overlapping grid cells between processes. Communication primitives are buried deep into the physical subroutines. Therefore, refactoring the codebase to leave `jax.jit` every time data needs to be communicated would severely break the control flow of the model and presumably incur a hefty performance loss (in addition to the cost of copying data from and to JAX). Through `mpi4jax`, it is possible to apply the JIT compiler to whole subroutines to avoid this entirely.

- netket...  **write me**


# Implementation

In essence, `mpi4jax` combines JAX's custom call mechanism with `mpi4py.libmpi` (which exposes MPI C primitives as Cython callables).


# Example: non-linear shallow water solver


# Acknowledgements

We thank all JAX developers in general and Matthew Johnson and Peter Hawkins in particular for their outstanding support on the many issues we opened.

DH received support from the Danish Hydrocarbon Research and Technology Centre (DHRTC).

# References
