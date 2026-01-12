import jax
import jax.numpy as jnp
import os
import sys

# Ensure immediate parent directory is in path to import math_art_jax
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from math_art_jax.engine.renderer import render_fn
from math_art_jax.warpers.polar import to_polar
from math_art_jax.utils.io import save_image


def fractal_shader(x, y, params):
    """
    A simple procedural shader for testing.
    """
    freq = params["freq"]
    r, theta = to_polar(x, y)

    # Create a flower-like pattern
    val = jnp.cos(theta * freq + r * 10.0)

    # Soft mask circle
    mask = jax.nn.sigmoid(10.0 * (1.0 - r))

    return val * mask


def main():
    print("Verifying math_art_jax engine...")

    width = 1024
    height = 1024
    params = {"freq": 12.0}

    print("Compiling and rendering...")
    # JIT compile the render function implicitly via vmap in render_fn?
    # Ideally we should jit the render_fn call if possible, or render_fn internals are jittable.
    # The render_fn constructs vmap, which is jittable.

    # To properly JIT everything, we can wrap the call
    jit_renderer = jax.jit(lambda p: render_fn(width, height, fractal_shader, p))

    image = jit_renderer(params)

    output_path = "verification_art.png"
    print(f"Saving to {output_path}...")
    save_image(image, output_path, cmap="magma")
    print("Done!")


if __name__ == "__main__":
    main()
