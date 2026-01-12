import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Callable, Any


def create_grid(
    width: int, height: int, xlim: tuple[float, float] = (-1.0, 1.0), ylim: tuple[float, float] = (-1.0, 1.0)
) -> tuple[Float[Array, "H W"], Float[Array, "H W"]]:
    """
    Generate coordinate grid.
    """
    x = jnp.linspace(xlim[0], xlim[1], width)
    y = jnp.linspace(ylim[0], ylim[1], height)
    # meshgrid returning xy indexing
    xv, yv = jnp.meshgrid(x, y)
    return xv, yv


def render_fn(
    width: int, height: int, pixel_shader: Callable[[float, float, Any], float], params: Any
) -> Float[Array, "H W"]:
    """
    Basic single-pass renderer using vmap.
    This assumes pixel_shader takes (x, y, params) and returns a float intensity.
    """
    xv, yv = create_grid(width, height)

    # We want to vmap the pixel_shader over the grid.
    # pixel_shader: (x, y, params) -> val

    # vmap over width then height
    # Or flatten?

    # Let's vectorize it efficiently.
    # vmap twice: first over x, then over y?
    # Or vmap over flattened arrays and reshape.

    # Assuming pixel_shader handles scalar inputs.

    # Using vmap to vectorize over the grid dimensions.
    # It constructs a function that takes arrays of x and y (and replicated params).

    # vmap(f, in_axes=(0, 0, None)) would map over the first dimension of x and y (Height),
    # but inside that we have Width.

    # Map over rows (H), then columns (W)
    # inner_vmap = jax.vmap(pixel_shader, in_axes=(0, 0, None)) # Maps over W
    # outer_vmap = jax.vmap(inner_vmap, in_axes=(0, 0, None))  # Maps over H

    # image = outer_vmap(xv, yv, params)

    # More robust: Flatten, vmap, reshape.
    xv_flat = xv.ravel()
    yv_flat = yv.ravel()

    shader_vmapped = jax.vmap(pixel_shader, in_axes=(0, 0, None))
    result_flat = shader_vmapped(xv_flat, yv_flat, params)

    return result_flat.reshape(height, width)
