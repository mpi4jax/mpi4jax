import jax


def flush():
    """Wait for all pending XLA operations"""
    # as suggested in jax#4335
    for device in jax.devices("cpu"):
        noop = jax.device_put(0, device=device) + 0
        noop.block_until_ready()
