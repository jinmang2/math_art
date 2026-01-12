import jax.numpy as jnp
from jaxtyping import Float, Array


def to_polar(x: Float[Array, "..."], y: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Convert Cartesian to Polar coordinates.
    Returns (r, theta).
    """
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    return r, theta


def from_polar(r: Float[Array, "..."], theta: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Convert Polar to Cartesian coordinates.
    Returns (x, y).
    """
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return x, y


def vortex_warp(
    x: Float[Array, "..."], y: Float[Array, "..."], strength: float = 1.0, decay: float = 1.0
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Apply a vortex swirl distortion.
    The angle of rotation increases as r decreases (or vice versa depending on effect).
    Here: theta' = theta + strength / (r^decay + epsilon)
    """
    r, theta = to_polar(x, y)
    theta_new = theta + strength / (r**decay + 1e-6)
    return from_polar(r, theta_new)


def spiral_log_warp(
    x: Float[Array, "..."], y: Float[Array, "..."], a: float = 0.2, b: float = 0.2
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Log-polar transformation for spiral patterns.
    u = log(r)
    v = theta
    Optionally mixed with linear rotation.
    """
    r, theta = to_polar(x, y)
    u = jnp.log(r + 1e-6)
    v = theta
    # Linear combination to create spirals in (u, v) space
    # Often visualization is done by mapping (u, v) back or using them as grid.
    # Here we return the transformed space coordinates U, V which might be treated as X, Y.
    return u * a - v * b, u * b + v * a
