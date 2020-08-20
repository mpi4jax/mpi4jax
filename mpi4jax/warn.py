import warnings


def warn_missing_omnistaging():
    import jax.config as jconfig

    if not jconfig.omnistaging_enabled:
        warnings.warn(
            "Without omnistaging, calls to jitted MPI routines might deadlock. "
            "Make sure to introduce data dependencies on all MPI calls, or use "
            "jax.config.enable_omnistaging() before importing mpi4jax."
        )
