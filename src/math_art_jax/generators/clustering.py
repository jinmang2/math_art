import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def generate_cluster_points(
    key: jax.random.KeyArray, n_points: int, center: tuple[float, float] = (0.0, 0.0), std: float = 0.1
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    """
    Generate a Gaussian cluster of points.
    """
    k1, k2 = jax.random.split(key)
    x = center[0] + std * jax.random.normal(k1, shape=(n_points,))
    y = center[1] + std * jax.random.normal(k2, shape=(n_points,))
    return x, y
