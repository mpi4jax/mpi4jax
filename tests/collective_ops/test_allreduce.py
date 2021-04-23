from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_allreduce():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, token = allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_jit():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = jax.jit(lambda x: allreduce(x, op=MPI.SUM)[0])(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_scalar():
    from mpi4jax import allreduce

    arr = 1
    _arr = 1

    res, token = allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_scalar_jit():
    from mpi4jax import allreduce

    arr = 1
    _arr = 1

    res = jax.jit(lambda x: allreduce(x, op=MPI.SUM)[0])(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_vmap():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = jax.vmap(lambda x: allreduce(x, op=MPI.SUM)[0], in_axes=0, out_axes=0)(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_vmap_jit():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = jax.jit(
        jax.vmap(lambda x: allreduce(x, op=MPI.SUM)[0], in_axes=0, out_axes=0)
    )(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_transpose():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    (res,) = jax.linear_transpose(lambda x: allreduce(x, op=MPI.SUM)[0], arr)(_arr)
    assert jnp.array_equal(_arr, res)


def test_allreduce_transpose_jit():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    def f(x):
        (res,) = jax.linear_transpose(lambda x: allreduce(x, op=MPI.SUM)[0], arr)(x)
        return res

    res = jax.jit(f)(arr)
    assert jnp.array_equal(_arr, res)


def test_allreduce_transpose2():
    # test transposing twice
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()
    _arr2 = arr.copy()

    def lt(y):
        return jax.linear_transpose(lambda x: allreduce(x, op=MPI.SUM)[0], arr)(y)[0]

    (res,) = jax.linear_transpose(lt, _arr)(_arr2)
    expected, _ = allreduce(_arr2, op=MPI.SUM)
    assert jnp.array_equal(expected, res)


def test_allreduce_transpose2_jit():
    # test transposing twice
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()
    _arr2 = arr.copy()

    def lt(y):
        return jax.linear_transpose(lambda x: allreduce(x, op=MPI.SUM)[0], arr)(y)[0]

    def f(x):
        (res,) = jax.linear_transpose(lt, _arr)(x)
        return res

    res = jax.jit(f)(_arr2)
    expected, _ = allreduce(_arr2, op=MPI.SUM)
    assert jnp.array_equal(expected, res)


def test_allreduce_grad():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, grad = jax.value_and_grad(lambda x: allreduce(x, op=MPI.SUM)[0].sum())(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)

    res, grad = jax.jit(
        jax.value_and_grad(lambda x: allreduce(x, op=MPI.SUM)[0].sum())
    )(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)

    def testfun(x):
        y, token = allreduce(x, op=MPI.SUM)
        z = x + 2 * y  # noqa: F841
        res, token = allreduce(x, op=MPI.SUM, token=token)
        return res.sum()

    res, grad = jax.jit(jax.value_and_grad(testfun))(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)


def test_allreduce_jvp():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, jvp = jax.jvp(lambda x: allreduce(x, op=MPI.SUM)[0], (arr,), (_arr,))

    expected, _ = allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(expected, res)
    expected, _ = allreduce(_arr, op=MPI.SUM)
    assert jnp.array_equal(expected, jvp)


def test_allreduce_vjp():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, vjp_fun = jax.vjp(lambda x: allreduce(x, op=MPI.SUM)[0], arr)
    (vjp,) = vjp_fun(_arr)

    expected, _ = allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(expected, res)
    assert jnp.array_equal(_arr, vjp)


def test_allreduce_chained():
    from mpi4jax import allreduce

    def foo(x):
        token = jax.lax.create_token()
        x1, token = allreduce(x, op=MPI.SUM, comm=comm, token=token)
        x2, token = allreduce(x, op=MPI.SUM, comm=comm, token=token)
        return x1 + x2

    res_t = jax.grad(foo)(0.0)

    expected = 2.0
    assert jnp.array_equal(expected, res_t)


def test_allreduce_chained_jit():
    from mpi4jax import allreduce

    def foo(x):
        token = jax.lax.create_token()
        x1, token = allreduce(x, op=MPI.SUM, comm=comm, token=token)
        x2, token = allreduce(x, op=MPI.SUM, comm=comm, token=token)
        return x1 + x2

    res_t = jax.jit(jax.grad(foo))(0.0)

    expected = 2.0
    assert jnp.array_equal(expected, res_t)
