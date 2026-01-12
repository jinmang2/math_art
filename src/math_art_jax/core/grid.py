import jax.numpy as jnp
from jaxtyping import Float, Array


def create_meshgrid(
    width: int, height: int, denominator: int | None = None
) -> tuple[Float[Array, "H W"], Float[Array, "H W"]]:
    """
    Creates a meshgrid for X and Y values, strictly matching the reference logic:

    x_vals = (np.arange(width) + 1 - width // 2) / (denominator or width // 2)
    y_vals = (height // 2 - np.arange(height)) / (denominator or height // 2)
    """

    denom_x = float(denominator if denominator is not None else width // 2)
    denom_y = float(denominator if denominator is not None else height // 2)

    # x: 1 to width (inclusive of 1? np.arange(width) is 0..W-1. +1 -> 1..W)
    x_indices = jnp.arange(width, dtype=jnp.float32)
    x_vals = (x_indices + 1.0 - (width // 2)) / denom_x

    # y: height//2 to ...
    y_indices = jnp.arange(height, dtype=jnp.float32)
    y_vals = ((height // 2) - y_indices) / denom_y

    # Create meshgrid
    # jnp.meshgrid defaults to 'xy' indexing (Cartesian) -> returns (W, H), (W, H) ?
    # No, meshgrid returns X (nrows, ncols) and Y (nrows, ncols).
    # If inputs are 1D arrays of shape N and M.
    # Output is M x N.

    xv, yv = jnp.meshgrid(x_vals, y_vals)

    return xv, yv
