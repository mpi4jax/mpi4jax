import jax


def flush():
    """Wait for all pending XLA operations"""
    # as suggested in jax#4335

    devices = jax.devices("cpu")

    # jax.devices("gpu") throws if no gpu available.
    try:
        devices.extend(jax.devices("gpu"))
    except RuntimeError:
        pass

    for device in devices:
        noop = jax.device_put(0, device=device) + 0
        noop.block_until_ready()
