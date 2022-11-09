from jax import abstract_arrays
from jax.core import Primitive

from .utils import  default_primitive_impl

from .decorators import translation_rule_cpu, translation_rule_gpu
from .jax_compat import register_abstract_eval

import jaxlib.mlir.ir as ir
from jax.interpreters import mlir
from jaxlib.mhlo_helpers import custom_call
import functools

# The Jax primitive
ascontiguousarray_p = Primitive("ascontiguousarray")  # Create the primitive
ascontiguousarray_impl = default_primitive_impl(ascontiguousarray_p)

def enforce_layout(function):
    """ Decorator that enforces that both input and output
    of the collective operations are in C contiguous order
    """
    @functools.wraps(function)
    def wrapped(x, *args, **kwargs):
        x = ascontiguousarray(x)
        out = function(x, *args, **kwargs)
        if function.__name__ == 'send':
            return out
        else:
            x, token = out
            x = ascontiguousarray(x)
            return x, token
    return wrapped

def ascontiguousarray(
    x,
):
    """ Enforces the layout of a given tensor to be C contiguous.
    If the array already has C layout, nothing happens.

    Arguments:
        x: Array input to potentially reorder
    
    Returns:
        Reordered array
    """
    return ascontiguousarray_p.bind(
            x,
        )

@translation_rule_gpu
def ascontiguousarray_xla_encode_gpu(ctx, x):
    dtype = ir.RankedTensorType(x.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))
    return [custom_call(
            b"ascontiguousarray",
        [dtype],
        [x],
        backend_config=None,
        operand_layouts=[layout],
        result_layouts=[layout],
        operand_output_aliases={0:0})]

@translation_rule_cpu
def ascontiguousarray_xla_encode_cpu(ctx, x):
    dtype = ir.RankedTensorType(x.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))
    return [custom_call(
            b"ascontiguousarray",
        [dtype],
        [x],
        backend_config=None,
        operand_layouts=[layout],
        result_layouts=[layout],
        operand_output_aliases={0:0})]     

# This function evaluates only the shapes during AST construction
def ascontiguousarray_abstract_eval(xs):
    return abstract_arrays.ShapedArray(xs.shape, xs.dtype)

ascontiguousarray_p.def_impl(ascontiguousarray_impl)
register_abstract_eval(ascontiguousarray_p, ascontiguousarray_abstract_eval)

mlir.register_lowering(
    ascontiguousarray_p,
    ascontiguousarray_xla_encode_gpu,
    platform='gpu')
mlir.register_lowering(
    ascontiguousarray_p,
    ascontiguousarray_xla_encode_cpu,
    platform='cpu')
