import jax.numpy as jnp
from jaxtyping import Float, Array


def safe_exp(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Prevents overflow/underflow in exponential calculations.
    Clips input to range [-1000, 700] before applying exp.
    Global DNA: Essential for handling high-power recursive functions.
    """
    return jnp.exp(jnp.clip(x, -1000.0, 700.0))


def soft_mask(x: Float[Array, "..."], k: float = 1000.0) -> Float[Array, "..."]:
    """
    Differentiable soft clipping/masking using sigmoid-like behavior.
    Approximates a step function as k -> infinity.
    Global DNA: Used for anti-aliased shape boundaries.
    """
    return 1.0 / (1.0 + safe_exp(-k * x))


def sharpen_op(x: Float[Array, "..."], power: float) -> Float[Array, "..."]:
    """
    High-power sharpening filter.
    e.g. cos(x)**2000 or exp(-|x|^p)
    """
    eps = 1e-12
    # Handling sign if needed, but typically used on positive magnitudes or cos^2
    return jnp.power(jnp.abs(x) + eps, power)
