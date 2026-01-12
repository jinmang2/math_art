import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def soft_mask(x: Float[Array, "..."], width: float = 0.01) -> Float[Array, "..."]:
    """
    Differentiable soft mask function.
    Returns values close to 1 when x < 0 and 0 when x > 0.
    The transition width is controlled by `width`.
    using sigmoid approximation: 1 / (1 + exp(x / width))
    """
    return jax.nn.sigmoid(-x / width)


def double_exp(x: Float[Array, "..."], p: float = 2000.0) -> Float[Array, "..."]:
    """
    High-power filter for extreme sharpness.
    Approximates the limit as p -> infinity to create hard edges from smooth functions.
    Commonly used as cos(x)**p or similar.
    Here we generalize to exp(-|x|^p) or similar shaping.
    For the specific art style of 'cos(k*pi*x)^2000', use direct power.

    This function implements a generalized sharpening:
    y = exp(-|x * p|)
    """
    return jnp.exp(-jnp.abs(x * p))


def safe_exp(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Safe exponential function with clipping to prevent overflow.
    Matches numpy reference implementation: np.exp(np.clip(x, -1000, 700))
    """
    return jnp.exp(jnp.clip(x, -1000.0, 700.0))


def f_val_transformer(val: Float[Array, "..."], exposure: float = 1.0) -> Float[Array, "..."]:
    """
    Global transformation function F(x) to map raw accumulation values to [0, 1].
    Typically a hyperbolic tangent or sigmoid to handle high dynamic range.
    """
    return jnp.tanh(val * exposure)
