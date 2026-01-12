import jax.numpy as jnp
from jaxtyping import Float, Array


def parametric_circle(t: Float[Array, "..."], r: float = 1.0) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    x = r * cos(t)
    y = r * sin(t)
    """
    return r * jnp.cos(t), r * jnp.sin(t)


def butterfly_curve(t: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Temple Fay's butterfly curve or similar.
    """
    # exp(cos(t)) - 2*cos(4t) - sin(t/12)^5
    r = jnp.exp(jnp.cos(t)) - 2 * jnp.cos(4 * t) - jnp.power(jnp.sin(t / 12.0), 5)
    return r * jnp.cos(t), r * jnp.sin(t)
