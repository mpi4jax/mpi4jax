# mpi4jax
![Tests](https://github.com/PhilipVinc/mpi4jax/workflows/Tests/badge.svg) [![codecov](https://codecov.io/gh/PhilipVinc/mpi4jax/branch/master/graph/badge.svg)](https://codecov.io/gh/PhilipVinc/mpi4jax)
 [![Conda Recipe](https://img.shields.io/badge/recipe-mpi4jax-green.svg)](https://anaconda.org/conda-forge/mpi4jax) 

MPI plugin for JAX, allowing MPI operations to be inserted in jitted blocks.

# Installation
You can Install `mpi4jax` through pip (see below) or conda (click on the badge)
```python
pip install mpi4jax                     # Pip
conda install -c conda-forge mpi4jax    # conda
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

## GPU Support

Mpi4jax also supports jax arrays stored on GPU devices. To use Jax on the GPU, make sure that your jaxlib is built with CUDA support by checking [jax Readme.md](https://github.com/google/jax#pip-installation). 
By default mpi4jax will activate a slow path, where GPU data is first transferred to the CPU, then the MPI operation is performed, and then the result is copied back to the GPU.

Direct GPU-to-GPU MPI operations are supported if your MPI distribution is built with CUDA support. 
If you know that is the case (or want to find out) export the following environment variable:
```python
export MPI4JAX_USE_CUDA_MPI=1
```

And data will not be copied to the CPU.
However, if your MPI library does not have CUDA support, python will most likely crash.


## Debugging

You can set the environment variable `MPI4JAX_DEBUG` to `1` to enable debug logging every time an MPI primitive is called from within a jitted function. You will then see messages like this:

```bash
$ MPI4JAX_DEBUG=1 mpirun -n 2 python send_recv.py
r0 | MPI_Send -> 1 with tag 0 and token 7fd7abc5f5c0
r1 | MPI_Recv <- 0 with tag -1 and token 7f9af7419ac0
```

## Contributors
- Filippo Vicentini @PhilipVinc
- Dion Häfner @dionhaefner
