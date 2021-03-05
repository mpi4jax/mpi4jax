# mpi4jax

![Tests](https://github.com/PhilipVinc/mpi4jax/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/PhilipVinc/mpi4jax/branch/master/graph/badge.svg)](https://codecov.io/gh/PhilipVinc/mpi4jax)
[![Conda Recipe](https://img.shields.io/badge/recipe-mpi4jax-green.svg)](https://anaconda.org/conda-forge/mpi4jax)
[![Documentation Status](https://readthedocs.org/projects/mpi4jax/badge/?version=latest)](https://mpi4jax.readthedocs.io/en/latest/?badge=latest)

MPI plugin for JAX, allowing MPI operations to be inserted in blocks compiled with `jax.jit`.

## Installation

You can install `mpi4jax` through `pip` or `conda`:

```bash
$ pip install mpi4jax                     # Pip
$ conda install -c conda-forge mpi4jax    # conda
```

## Supported operations

- Bcast
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
b, token = mpi4jax.Allreduce(a, op=MPI.SUM, comm=comm)
b_jit, token = jax.jit(lambda x: mpi4jax.Allreduce(x, op=MPI.SUM, comm=comm))(a)
```

## GPU Support

`mpi4jax` also supports JAX arrays stored in GPU memory. To use JAX on the GPU, make sure that your `jaxlib` is [built with CUDA support](https://github.com/google/jax#pip-installation).

Currently, we cannot detect whether MPI was built with CUDA support. Therefore, by default, `mpi4jax` will not read directly from GPU memory, but instead copy to the CPU and back.

If you are certain that the underlying MPI library was built with CUDA support, you can set the following environment variable:

```bash
$ export MPI4JAX_USE_CUDA_MPI=1
```

Data will then be copied directly from GPU to GPU. If your MPI library does not have CUDA support, you will receive a segmentation fault when trying to access GPU memory.

## Contributing

We use pre-commit hooks to enforce a common code format. To install them, just run:

```bash
$ pip install pre-commit
$ pre-commit install
```

## Debugging

You can set the environment variable `MPI4JAX_DEBUG` to `1` to enable debug logging every time an MPI primitive is called from within a jitted function. You will then see messages like this:

```bash
$ MPI4JAX_DEBUG=1 mpirun -n 2 python send_recv.py
r0 | MPI_Send -> 1 with tag 0 and token 7fd7abc5f5c0
r1 | MPI_Recv <- 0 with tag -1 and token 7f9af7419ac0
```

## Contributors

- Filippo Vicentini [@PhilipVinc](https://github.com/PhilipVinc)
- Dion Häfner [@dionhaefner](https://github.com/dionhaefner)
