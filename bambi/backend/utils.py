import aesara.tensor as at
import pymc as pm


def get_distribution(dist):
    """Return a PyMC distribution."""
    if isinstance(dist, str):
        if hasattr(pm, dist):
            return getattr(pm, dist)

    return dist


def has_hyperprior(kwargs):
    """Determines if a Prior has an hyperprior"""
    return (
        "sigma" in kwargs
        and "observed" not in kwargs
        and isinstance(kwargs["sigma"], at.TensorVariable)
    )
