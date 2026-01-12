import jax.numpy as jnp
from jaxtyping import Float, Array


def apply_perspective(
    x: Float[Array, "..."], y: Float[Array, "..."], z: Float[Array, "..."], fov: float = 1.0
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Simple 3D perspective projection onto 2D plane.
    x_proj = x / (z + fov)
    y_proj = y / (z + fov)
    Assumes camera at (0, 0, -fov) looking at +z ?
    Or usually z is depth.
    Let's assume standard z division.
    """
    factor = 1.0 / (z + fov + 1e-6)
    return x * factor, y * factor
