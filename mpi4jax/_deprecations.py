import warnings
import functools
from ._src import allreduce, bcast, send, recv, sendrecv


def deprecated_new_name(message):
    def deprecated_decorator(func):
        @functools.wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                """{} has been renamed to {} and deprecated.
                It will be removed in the next minor version.
                """.format(
                    func.__name__, message
                ),
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return deprecated_func

    return deprecated_decorator


@deprecated_new_name("allreduce")
def Allreduce(*args, **kwargs):
    return allreduce(*args, **kwargs)


@deprecated_new_name("bcast")
def Bcast(*args, **kwargs):
    return bcast(*args, **kwargs)


@deprecated_new_name("recv")
def Recv(*args, **kwargs):
    return recv(*args, **kwargs)


@deprecated_new_name("send")
def Send(*args, **kwargs):
    return send(*args, **kwargs)


@deprecated_new_name("sendrecv")
def Sendrecv(*args, **kwargs):
    return sendrecv(*args, **kwargs)
