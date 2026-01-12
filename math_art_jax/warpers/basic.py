import jax.numpy as jnp
from jaxtyping import Float, Array


def scale(
    x: Float[Array, "..."], y: Float[Array, "..."], sx: float, sy: float
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Scale coordinates by sx, sy.
    """
    return x * sx, y * sy


def rotate(
    x: Float[Array, "..."], y: Float[Array, "..."], theta: float
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Rotate coordinates by theta (radians).
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    nx = x * c - y * s
    ny = x * s + y * c
    return nx, ny


def translate(
    x: Float[Array, "..."], y: Float[Array, "..."], dx: float, dy: float
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Translate coordinates by dx, dy.
    """
    return x - dx, y - dy
