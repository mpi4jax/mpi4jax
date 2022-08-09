import jax


def flush(platform):
    """Wait for all pending XLA operations"""
    # TODO: replace with jax.effects_barrier() once we drop JAX<0.3.15
    devices = jax.devices(platform)

    for device in devices:
        # as suggested in jax#4335
        noop = jax.device_put(0, device=device) + 0
        noop.block_until_ready()
