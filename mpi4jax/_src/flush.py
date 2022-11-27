import jax


def flush():
    """Wait for all pending XLA operations"""
    jax.effects_barrier()
