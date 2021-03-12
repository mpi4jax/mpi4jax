import pytest


def test_enforce_types():
    from types import FunctionType

    from mpi4jax._src.validation import enforce_types

    @enforce_types(y=(int, str), z=FunctionType)
    def foo(x, y, z):
        pass

    # ok
    foo(1, 2, lambda x: x)
    foo("test", "test", lambda x: x)

    # not ok
    with pytest.raises(TypeError) as exc:
        foo(1, lambda x: x, lambda x: x)

    assert "expected: ['int', 'str'], got: <class 'function'>" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        foo(1, 2, 3)

    assert "expected: function, got: <class 'int'>" in str(exc.value)


def test_enforce_types_generic():
    import numpy as np

    from mpi4jax._src.validation import enforce_types

    @enforce_types(x=np.integer)
    def foo(x):
        pass

    # ok
    foo(1)
    foo(np.uint64(1))
    foo(np.int32(1))

    # not ok
    with pytest.raises(TypeError) as exc:
        foo(True)

    assert "expected: integer, got: <class 'bool'>" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        foo(1.2)

    assert "expected: integer, got: <class 'float'>" in str(exc.value)


def test_enforce_types_invalid_args():
    from mpi4jax._src.validation import enforce_types

    def foo(x):
        pass

    with pytest.raises(ValueError) as exc:
        enforce_types(a=int)(foo)

    assert 'got unexpected argument "a"' in str(exc.value)


def test_enforce_types_tracer():
    import jax

    from mpi4jax._src.validation import enforce_types

    @enforce_types(x=int)
    def foo(x):
        pass

    # ok
    jax.jit(foo, static_argnums=(0,))(0)

    # not ok
    with pytest.raises(TypeError) as exc:
        jax.jit(foo)(0)

    assert "abstract tracer was passed" in str(exc.value)
