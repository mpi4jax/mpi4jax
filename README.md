# mpi4jax
![Tests](https://github.com/PhilipVinc/mpi4jax/workflows/Tests/badge.svg)

MPI plugin for JAX, allowing MPI operations to be inserted in jitted blocks.

## Installation

```python
pip install mpi4jax
```

## Supported operations

- Send
- Recv
- Sendrecv
- Allreduce

## Usage

```python
from mpi4py import MPI
import jax
import mpi4jax

comm = MPI.COMM_WORLD
a = jax.numpy.ones(5,4)
b = mpi4jax.Allreduce(a, op=MPI.SUM, comm=comm)
b_jit = jax.jit(lambda x: mpi4jax.Allreduce(x, op=MPI.SUM, comm=comm))(a)
```

## Contributors
- Filippo Vicentini @PhilipVinc
- Dion Häfner @dionhaefner
