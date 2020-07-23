# mpi4jax

![Tests](https://github.com/PhilipVinc/mpi4jax/workflows/Tests/badge.svg)

Prototype of an MPI plugin for JAX, allowing MPI collective 
operations to be inserted in jitted blocks and be traced through by AD.

# Installation
```python
pip install https://github.com/PhilipVinc/mpi4jax
```

# Supported operations

- Allreduce

# Contributors
Filippo Vicentini @PhilipVinc
Dion Häfner @dionhaefner 