import jax


def flush():
    """Wait for all pending XLA operations"""
    # as suggested in jax#4335
    noop = jax.device_put(0, device='cpu') + 0
    noop.block_until_ready()
