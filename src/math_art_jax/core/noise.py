import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Key


def white_noise(key: Key, shape: tuple) -> Float[Array, "..."]:
    """
    Basic white noise generator.
    """
    return jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0)


def harmonic_sum(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    freqs: Float[Array, "N"],
    amps: Float[Array, "N"],
    phases: Float[Array, "N"],
) -> Float[Array, "..."]:
    """
    Sum of harmonics for procedural texturing.
    val = sum(amp[i] * sin(freq[i] * x + phase[i] * y))
    """
    # Reshape for broadcasting
    # x, y: [H, W]
    # freqs, amps, phases: [N]

    # We want to sum over the N dimension.
    # Expand x, y to [H, W, 1]
    x_ex = jnp.expand_dims(x, axis=-1)
    y_ex = jnp.expand_dims(y, axis=-1)

    # Calculate terms: amp * sin(freq * (x + y) + ...?)
    # Usually noise is spatial. Let's assume separated components or combined.
    # Simple harmonic interference: sin(freq*x) * sin(freq*y) or similar.
    # Let's implement a directional harmonic: sin(freq * (cos(phase)*x + sin(phase)*y))

    kx = jnp.cos(phases) * freqs  # [N]
    ky = jnp.sin(phases) * freqs  # [N]

    arguments = x_ex * kx + y_ex * ky  # [H, W, N]
    components = amps * jnp.sin(arguments)  # [H, W, N]

    return jnp.sum(components, axis=-1)
