import jax
import jax.numpy as jnp

from mpi4py import MPI
from mpi4jax import allreduce

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
    res, _ = allreduce(x, op=MPI.SUM, comm=MPI.COMM_WORLD)
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
    return A_local.T @ transpose(allreduce_sum, y_global)(y_global)


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


def test_matvec_jit():
    res = jax.jit(mv)(x_local)
    assert jnp.allclose(Ax, res)


def test_matvecT():
    res = mvT(y)
    assert jnp.allclose(ATy_local, res)


def test_matvecT_jit():
    res = jax.jit(mvT)(y)
    assert jnp.allclose(ATy_local, res)


def test_matvec_transpose():
    lt = transpose(mv, x_local)
    res = lt(y)
    assert jnp.allclose(ATy_local, res)


def test_matvec_transpose_jit():
    lt = transpose(mv, x_local)
    res = jax.jit(lt)(y)
    assert jnp.allclose(ATy_local, res)


def test_matvecT_transpose():
    ltT = transpose(mvT, y)
    res = ltT(x_local)
    assert jnp.allclose(Ax, res)


def test_matvecT_transpose_jit():
    ltT = transpose(mvT, y)
    res = jax.jit(ltT)(x_local)
    assert jnp.allclose(Ax, res)


def test_matvec_transpose2():
    lt = transpose(mv, x_local)
    ltlt = transpose(lt, y)
    res = ltlt(x_local)
    assert jnp.allclose(Ax, res)


def test_matvec_transpose2_jit():
    lt = transpose(mv, x_local)
    ltlt = transpose(lt, y)
    res = jax.jit(ltlt)(x_local)
    assert jnp.allclose(Ax, res)


def test_matvecT_transpose2():
    ltT = transpose(mvT, y)
    ltltT = transpose(ltT, x_local)
    res = ltltT(y)
    assert jnp.allclose(ATy_local, res)


def test_matvecT_transpose2_jit():
    ltT = transpose(mvT, y)
    ltltT = transpose(ltT, x_local)
    res = jax.jit(ltltT)(y)
    assert jnp.allclose(ATy_local, res)


def test_matvec_transpose3():
    lt = transpose(mv, x_local)
    ltlt = transpose(lt, y)
    ltltlt = transpose(ltlt, x_local)
    res = ltltlt(y)
    assert jnp.allclose(ATy_local, res)


def test_matvec_transpose3_jit():
    lt = transpose(mv, x_local)
    ltlt = transpose(lt, y)
    ltltlt = transpose(ltlt, x_local)
    res = jax.jit(ltltlt)(y)
    assert jnp.allclose(ATy_local, res)


def test_matvecT_transpose3():
    ltT = transpose(mvT, y)
    ltltT = transpose(ltT, x_local)
    ltltltT = transpose(ltltT, y)
    res = ltltltT(x_local)
    assert jnp.allclose(Ax, res)


def test_matvecT_transpose3_jit():
    ltT = transpose(mvT, y)
    ltltT = transpose(ltT, x_local)
    ltltltT = transpose(ltltT, y)
    res = jax.jit(ltltltT)(x_local)
    assert jnp.allclose(Ax, res)


def test_matvec_jvp():
    res, jvp = jax.jvp(mv, (x_local,), (v_local,))
    assert jnp.allclose(Ax, res)
    assert jnp.allclose(Av, jvp)


def test_matvec_jvp_jit():
    res, jvp = jax.jit(lambda x, v: jax.jvp(mv, (x,), (v,)))(x_local, v_local)
    assert jnp.allclose(Ax, res)
    assert jnp.allclose(Av, jvp)


def test_matvecT_jvp():
    res, jvp = jax.jvp(mvT, (y,), (vprime,))
    assert jnp.allclose(ATy_local, res)
    assert jnp.allclose(ATvprime_local, jvp)


def test_matvecT_jvp_jit():
    res, jvp = jax.jit(lambda y, v: jax.jvp(mvT, (y,), (v,)))(y, vprime)
    assert jnp.allclose(ATy_local, res)
    assert jnp.allclose(ATvprime_local, jvp)


def test_matvec_vjp():
    res, vjp_fun = jax.vjp(mv, x_local)
    (vjp,) = vjp_fun(vprime)
    assert jnp.allclose(Ax, res)
    assert jnp.allclose(ATvprime_local, vjp)


def test_matvec_vjp_jit():
    def f(x, v):
        res, vjp_fun = jax.vjp(mv, x)
        (vjp,) = vjp_fun(v)
        return res, vjp

    res, vjp = jax.jit(f)(x_local, vprime)
    assert jnp.allclose(Ax, res)
    assert jnp.allclose(ATvprime_local, vjp)


def test_matvecT_vjp():
    res, vjp_fun = jax.vjp(mvT, y)
    (jvp,) = vjp_fun(v_local)
    assert jnp.allclose(ATy_local, res)
    assert jnp.allclose(Av, jvp)


def test_matvecT_vjp_jit():
    def f(y, v):
        res, vjp_fun = jax.vjp(mvT, y)
        (jvp,) = vjp_fun(v)
        return res, jvp

    res, jvp = jax.jit(f)(y, v_local)
    assert jnp.allclose(ATy_local, res)
    assert jnp.allclose(Av, jvp)
