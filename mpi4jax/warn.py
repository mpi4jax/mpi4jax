import warnings

omnistaging_warning_enabled = True


def disable_omnistaging_warning():
    """
    Disables the warning raised when jax omnistaging is disabled.

    """
    global omnistaging_warning_enabled
    omnistaging_warning_enabled = False


def warn_missing_omnistaging():
    import jax.config as jconfig

    if omnistaging_warning_enabled and not jconfig.omnistaging_enabled:
        warnings.warn(
            "Without omnistaging, calls to jitted MPI routines might deadlock. "
            "Make sure to introduce data dependencies on all MPI calls, or use "
            "jax.config.enable_omnistaging() before importing mpi4jax."
        )
