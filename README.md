# mpi4jax
Prototype of an MPI plugin for JAX (work in progress), allowing MPI collective 
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