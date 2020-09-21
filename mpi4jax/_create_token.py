"""
This file defines a create_token operation analogous to the jax.lax one,
but for which we also define the gradient, so that it plays nice with AD.
"""
from jax.abstract_arrays import abstract_token
from jax.core import Primitive
from jax.interpreters import ad, xla
from jax.lax import zeros_like_array
from jax.lib import xla_client
from jax.util import partial


def create_token(x):
    """Creates an XLA token value with no preconditions for sequencing effects.
    This is a mpi4jax customized version, which behaves as the jax one but it
    is also possible to compute the gradient of it.

    Experimental.

    Args:
      x: a dummy argument used to tie the CreateToken operator into a trace. The
         value of `x` is ignored.
    """
    # x is a dummy argument used to tie the operator into a trace.
    return create_token_p.bind(x)


create_token_p = Primitive("create_token_mpi4jax")
create_token_p.def_impl(partial(xla.apply_primitive, create_token_p))
create_token_p.def_abstract_eval(lambda _: abstract_token)
xla.translations[create_token_p] = lambda c, _: xla_client.ops.CreateToken(c)


def create_token_value_and_jvp(in_args, tan_args):
    (x,) = in_args
    res = create_token(x)
    jvp = zeros_like_array(x)
    return (res, jvp)


ad.primitive_jvps[create_token_p] = create_token_value_and_jvp
