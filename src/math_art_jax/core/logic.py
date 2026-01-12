import jax.numpy as jnp
from jaxtyping import Float, Array


def soft_and(a: Float[Array, "..."], b: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Soft logical AND (multiplication).
    Assumes inputs are in [0, 1].
    """
    return a * b


def soft_or(a: Float[Array, "..."], b: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Soft logical OR.
    a + b - a*b (probabilistic OR) or max(a, b)
    Using max is often sharper for Constructive Solid Geometry (CSG).
    """
    return jnp.maximum(a, b)


def soft_not(a: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Soft logical NOT.
    """
    return 1.0 - a


def soft_xor(a: Float[Array, "..."], b: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Soft logical XOR.
    |a - b|
    """
    return jnp.abs(a - b)
