import jax
from mpi4jax._src.jax_compat import versiontuple

if versiontuple(jax.__version__) <= versiontuple("0.4.10"):
    raise ImportError(
        "mpi4jax.experimental is not compatible with JAX versions <= 0.4.10 "
    )

from .tokenizer import auto_tokenize

__all__ = [
    "auto_tokenize",
]
