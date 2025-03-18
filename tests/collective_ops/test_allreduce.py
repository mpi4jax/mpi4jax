from mpi4py import MPI

import jax
import jax.numpy as jnp

import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def test_allreduce():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_jit():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = jax.jit(lambda x: allreduce(x, op=MPI.SUM))(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_scalar():
    from mpi4jax import allreduce

    arr = 1
    _arr = 1

    res = allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_scalar_jit():
    from mpi4jax import allreduce

    arr = 1
    _arr = 1

    res = jax.jit(lambda x: allreduce(x, op=MPI.SUM))(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_vmap():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = jax.vmap(lambda x: allreduce(x, op=MPI.SUM), in_axes=0, out_axes=0)(arr)
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_vmap_jit():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res = jax.jit(jax.vmap(lambda x: allreduce(x, op=MPI.SUM), in_axes=0, out_axes=0))(
        arr
    )
    assert jnp.array_equal(res, arr * size)
    assert jnp.array_equal(_arr, arr)


def test_allreduce_transpose():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    (res,) = jax.linear_transpose(lambda x: allreduce(x, op=MPI.SUM), arr)(_arr)
    assert jnp.array_equal(_arr, res)


def test_allreduce_transpose_jit():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    def f(x):
        (res,) = jax.linear_transpose(lambda x: allreduce(x, op=MPI.SUM), arr)(x)
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
        return jax.linear_transpose(lambda x: allreduce(x, op=MPI.SUM), arr)(y)[0]

    (res,) = jax.linear_transpose(lt, _arr)(_arr2)
    expected = allreduce(_arr2, op=MPI.SUM)
    assert jnp.array_equal(expected, res)


def test_allreduce_transpose2_jit():
    # test transposing twice
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()
    _arr2 = arr.copy()

    def lt(y):
        return jax.linear_transpose(lambda x: allreduce(x, op=MPI.SUM), arr)(y)[0]

    def f(x):
        (res,) = jax.linear_transpose(lt, _arr)(x)
        return res

    res = jax.jit(f)(_arr2)
    expected = allreduce(_arr2, op=MPI.SUM)
    assert jnp.array_equal(expected, res)


def test_allreduce_grad():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, grad = jax.value_and_grad(lambda x: allreduce(x, op=MPI.SUM).sum())(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)

    res, grad = jax.jit(jax.value_and_grad(lambda x: allreduce(x, op=MPI.SUM).sum()))(
        arr
    )
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)

    def testfun(x):
        y = allreduce(x, op=MPI.SUM)
        z = x + 2 * y  # noqa: F841
        res = allreduce(x, op=MPI.SUM)
        return res.sum()

    res, grad = jax.jit(jax.value_and_grad(testfun))(arr)
    assert jnp.array_equal(res, arr.sum() * size)
    assert jnp.array_equal(_arr, grad)


def test_allreduce_jvp():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, jvp = jax.jvp(lambda x: allreduce(x, op=MPI.SUM), (arr,), (_arr,))

    expected = allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(expected, res)
    expected = allreduce(_arr, op=MPI.SUM)
    assert jnp.array_equal(expected, jvp)


def test_allreduce_vjp():
    from mpi4jax import allreduce

    arr = jnp.ones((3, 2))
    _arr = arr.copy()

    res, vjp_fun = jax.vjp(lambda x: allreduce(x, op=MPI.SUM), arr)
    (vjp,) = vjp_fun(_arr)

    expected = allreduce(arr, op=MPI.SUM)
    assert jnp.array_equal(expected, res)
    assert jnp.array_equal(_arr, vjp)


def test_allreduce_chained():
    from mpi4jax import allreduce

    def foo(x):
        x1 = allreduce(x, op=MPI.SUM, comm=comm)
        print(x1)
        x2 = allreduce(x, op=MPI.SUM, comm=comm)
        print(x2)
        return x1 + x2

    res_t = jax.grad(foo)(0.0)

    expected = 2.0
    assert jnp.array_equal(expected, res_t)


def test_allreduce_chained_jit():
    from mpi4jax import allreduce

    def foo(x):
        x1 = allreduce(x, op=MPI.SUM, comm=comm)
        x2 = allreduce(x, op=MPI.SUM, comm=comm)
        return x1 + x2

    res_t = jax.jit(jax.grad(foo))(0.0)

    expected = 2.0
    np.testing.assert_allclose(expected, res_t)


def test_custom_vjp():
    from mpi4jax import allreduce

    # define an arbitrary function with custom_vjp
    @jax.custom_vjp
    def f(x, y):
        r = jnp.sin(x) * y
        r = r.sum()
        return allreduce(r, op=MPI.SUM)

    def f_fwd(x, y):
        # Returns primal output and residuals to be used in backward pass by f_bwd.
        return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
        g = allreduce(g, op=MPI.SUM)
        cos_x, sin_x, y = res  # Gets residuals computed in f_fwd
        return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd)

    # check that it does not crash
    _ = jax.jit(f)(jnp.ones(3), jnp.ones(3) * 2)
    _ = jax.jit(jax.grad(f))(jnp.ones(3), jnp.ones(3) * 2)


def test_advanced_jvp():
    """
    This is an integration test checking that the effects introduced by mpi4jax
    are correctly handled by jax in a complex `custom_vjp` rule computing a
    `jax.grad` inside of the rule itself.

    This code essentially computes the mathematically correct gradient of an
    expectation value of a function over a probability distribution.
    It is taken from netket.jax.expect
    https://github.com/netket/netket/blob/master/netket/jax/_expect.py
    """
    from mpi4jax import allreduce
    from functools import partial

    def expect(
        log_pdf,
        expected_fun,
        pars,
        x,
        *expected_fun_args,
        n_chains,
    ):
        return _expect(n_chains, log_pdf, expected_fun, pars, x, *expected_fun_args)

    @partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
    def _expect(n_chains, log_pdf, expected_fun, pars, x, *expected_fun_args):
        L_x = expected_fun(pars, x, *expected_fun_args).reshape((n_chains, -1))
        return allreduce(L_x.mean(), op=MPI.SUM) / MPI.COMM_WORLD.Get_size()

    def _expect_fwd(n_chains, log_pdf, expected_fun, pars, x, *expected_fun_args):
        L_x = expected_fun(pars, x, *expected_fun_args)
        L_x_r = L_x.reshape((n_chains, -1))
        dL_x = allreduce(L_x_r.mean(), op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        ΔL_x = L_x - dL_x
        return dL_x, (pars, x, expected_fun_args, ΔL_x)

    def _expect_bwd(n_chains, log_pdf, expected_fun, residuals, dout):
        pars, x, cost_args, ΔL_x = residuals
        dL = dout

        def f(pars, x, *cost_args):
            log_p = log_pdf(pars, x)
            term1 = jax.vmap(jnp.multiply)(ΔL_x, log_p)
            term2 = expected_fun(pars, x, *cost_args)
            out = (
                allreduce(jnp.mean(term1 + term2, axis=0), op=MPI.SUM)
                / MPI.COMM_WORLD.Get_size()
            )
            out = out.sum()
            return out

        aa, pb = jax.vjp(f, pars, x, *cost_args)
        grad_f = pb(dL)
        return grad_f

    _expect.defvjp(_expect_fwd, _expect_bwd)

    def log_pdf(w, x):
        return jnp.sum(x.dot(w), axis=-1)

    def expected_fun(w, x, *args):
        return jnp.exp(jnp.sum(x.dot(w), axis=-1)) - 2

    w = jax.random.normal(jax.random.PRNGKey(3), (4, 8))
    x = jax.random.normal(jax.random.PRNGKey(3), (16, 4))

    expect(log_pdf, expected_fun, w, x, None, n_chains=16)
    O, vjpfun = jax.vjp(
        lambda w: expect(log_pdf, expected_fun, w, x, None, n_chains=16), w
    )
    vjpfun(jnp.ones_like(O))
