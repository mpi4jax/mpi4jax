import inspect
import functools

from jax.core import Tracer


def enforce_types(**types):
    """Decorator that enforces given argument types at runtime.

    Throws a TypeError if an invalid type is passed.

    Example:

        >>> @enforce_types(
        ...     foo=(int, type(None)),
        ...     bar=str
        ... )
        ... def func(x, foo, bar):
        ...     print(bar)

        >>> func(1, 2, 'hi there!')
        hi there!

        >>> func(1, 2, 3)
        TypeError: func got unexpected type for argument "bar" (expected: str, got: <class 'int'>)

    """

    def decorator(function):
        func_name = function.__name__
        func_sig = inspect.signature(function)

        for t in types:
            # make sure given types are actually parameters of decorated function
            if t not in func_sig.parameters:
                raise ValueError(
                    f"enforce_types decorator for {func_name} got unexpected argument {t}"
                )

            # make sure types are iterable
            if not isinstance(types[t], (tuple, list)):
                types[t] = (types[t],)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            # parse passed args
            bound_args = func_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # check one by one
            for arg, val in bound_args.arguments.items():
                if arg not in types:
                    continue

                arg_types = types[arg]

                if not isinstance(val, arg_types):
                    readable_arg_types = [t.__qualname__ for t in arg_types]
                    if len(readable_arg_types) == 1:
                        readable_arg_types = readable_arg_types[0]

                    extra_message = ""
                    if isinstance(val, Tracer):
                        extra_message = (
                            "\n\nAn abstract tracer was passed where a concrete value is expected. "
                            "Try using the static_argnums argument of jax.jit."
                        )

                    raise TypeError(
                        f'{func_name} got unexpected type for argument "{arg}" '
                        f"(expected: {readable_arg_types}, got: {type(val)})."
                        f"{extra_message}"
                    )

            return function(*args, **kwargs)

        return wrapped

    return decorator
