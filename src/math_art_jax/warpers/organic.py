import jax.numpy as jnp
from jaxtyping import Float, Array


def sinusoidal_warp(
    x: Float[Array, "..."], y: Float[Array, "..."], freq: float = 5.0, amp: float = 0.1
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Add sinusoidal waviness to coordinates.
    """
    nx = x + amp * jnp.sin(y * freq)
    ny = y + amp * jnp.sin(x * freq)
    return nx, ny


def polynomial_warp(
    x: Float[Array, "..."], y: Float[Array, "..."], k: float = 0.5
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Simple polynomial distortion, e.g. barrel/pincushion.
    """
    r2 = x**2 + y**2
    factor = 1.0 + k * r2
    return x * factor, y * factor


# --- Atomic Breakdown: Strawberry Phase Warps ---


def strawberry_p_warp(x: Float[Array, "..."], y: Float[Array, "..."], s: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Computes the Phase P(x, y, s) for Strawberries.
    Represents a specific angular distortion based on layer 's'.
    """
    inner_tan = jnp.tan(2 * jnp.sin(5 * s) * x - 2 * jnp.cos(5 * s) * y + 3 * jnp.cos(5 * s))
    correction = (3 * jnp.cos(14 * x - 19 * y + 5 * s)) / 200.0
    return jnp.arctan(inner_tan + correction)


def strawberry_q_warp(x: Float[Array, "..."], y: Float[Array, "..."], s: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Computes the Phase Q(x, y, s) for Strawberries.
    Represents the secondary orthogonal angular distortion.
    """
    inner_tan = jnp.tan(2 * (jnp.cos(5 * s) * x + jnp.sin(5 * s) * y) + 2 * jnp.cos(4 * s))
    correction = (3 * jnp.cos(18 * x + 15 * y + 4 * s)) / 200.0
    return jnp.arctan(inner_tan + correction)
