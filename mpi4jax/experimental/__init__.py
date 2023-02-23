import jax
from mpi4jax._src.jax_compat import versiontuple

if versiontuple(jax.__version__) >= versiontuple("0.4.4"):
    raise ImportError(
        "mpi4jax.experimental is not yet compatible with JAX versions >= 0.4.4. "
        "Please use jax<=0.4.3 for now."
    )

from .tokenizer import auto_tokenize

__all__ = [
    "auto_tokenize",
]
