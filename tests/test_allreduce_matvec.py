#!/usr/bin/env python

"""
Run with

$ mpirun -n <nproc> python -m pytest .
"""

# don't hog memory if running on GPU
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax  # noqa: E402
import jax.config  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.enable_omnistaging()

from mpi4py import MPI  # noqa: E402
from mpi4jax import Allreduce  # noqa: E402

print(MPI.get_vendor())

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

print("MPI rank = ", rank)
print("MPI size = ", size)


seed = 123
m, n = 5, 60

k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(seed), 5)
A = jax.random.uniform(k1, (m, n))
x = jax.random.uniform(k2, (n,))
y = jax.random.uniform(k3, (m,))
v = jax.random.uniform(k4, (n,))
vprime = jax.random.uniform(k5, (m,))


assert n % size == 0
n_local = n // size
start_local = n_local * rank
end_local = start_local + n_local

A_local = A[:, start_local:end_local]
x_local = x[start_local:end_local]
v_local = v[start_local:end_local]


# expected results:
Ax = A @ x
ATy_local = (A.T @ y)[start_local:end_local]

Av = A @ v
ATvprime_local = (A.T @ vprime)[start_local:end_local]


def allreduce_sum(x):
    res, token = Allreduce(x, op=MPI.SUM, comm=MPI.COMM_WORLD)
    return res


# this is just an identity
def allreduce_sumT(y):
    res, token = Allreduce(y, op=MPI.SUM, comm=MPI.COMM_WORLD, _transpose=True)
    return res


# x is distributed across ranks
# cols of A are distributed across ranks
# result is same on all ranks
def matvec_mpi(A_local, x_local):
    return allreduce_sum(A_local @ x_local)


# y is same on all ranks
# rows of A.T are distributed across ranks
# result is distributed across ranks
def matvec_transpose_mpi(A_local, y_global):
    return A_local.T @ allreduce_sumT(y_global)


def mv(x_local):
    return matvec_mpi(A_local, x_local)


def mvT(y):
    return matvec_transpose_mpi(A_local, y)


def transpose(f, x):
    def fT(y):
        return jax.linear_transpose(f, x)(y)[0]

    return fT


# tests


def test_matvec():
    res = mv(x_local)
    assert jnp.allclose(Ax, res)


def test_matvecT():
    res = mvT(y)
    assert jnp.allclose(ATy_local, res)


def test_matvec_transpose():
    lt = transpose(mv, x_local)
    res = lt(y)
    assert jnp.allclose(ATy_local, res)


def test_matvecT_transpose():
    ltT = transpose(mvT, y)
    res = ltT(x_local)
    assert jnp.allclose(Ax, res)


def test_matvec_transpose2():
    lt = transpose(mv, x_local)
    ltlt = transpose(lt, y)
    res = ltlt(x_local)
    assert jnp.allclose(Ax, res)


def test_matvecT_transpose2():
    ltT = transpose(mvT, y)
    ltltT = transpose(ltT, x_local)
    res = ltltT(y)
    assert jnp.allclose(ATy_local, res)


def test_matvec_transpose3():
    lt = transpose(mv, x_local)
    ltlt = transpose(lt, y)
    ltltlt = transpose(ltlt, x_local)
    res = ltltlt(y)
    assert jnp.allclose(ATy_local, res)


def test_matvecT_transpose3():
    ltT = transpose(mvT, y)
    ltltT = transpose(ltT, x_local)
    ltltltT = transpose(ltltT, y)
    res = ltltltT(x_local)
    assert jnp.allclose(Ax, res)


def test_matvec_jvp():
    res, jvp = jax.jvp(mv, (x_local,), (v_local,))
    assert jnp.allclose(Ax, res)
    assert jnp.allclose(Av, jvp)


def test_matvecT_jvp():
    res, jvp = jax.jvp(mvT, (y,), (vprime,))
    assert jnp.allclose(ATy_local, res)
    assert jnp.allclose(ATvprime_local, jvp)


def test_matvec_vjp():
    res, vjp_fun = jax.vjp(mv, x_local)
    (vjp,) = vjp_fun(vprime)
    assert jnp.allclose(Ax, res)
    assert jnp.allclose(ATvprime_local, vjp)


def test_matvecT_vjp():
    res, vjp_fun = jax.vjp(mvT, y)
    (jvp,) = vjp_fun(v_local)
    assert jnp.allclose(ATy_local, res)
    assert jnp.allclose(Av, jvp)
