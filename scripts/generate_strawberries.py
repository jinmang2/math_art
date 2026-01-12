import jax
import jax.numpy as jnp
import json
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
# (Assuming script is run from project root, 'src' is in path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Enable x64 precision for strict reproduction of art formulas
jax.config.update("jax_enable_x64", True)

from recipes.strawberries import strawberry_pipeline


def main():
    print("Loading config...")
    # Assume running from project root or scripts/
    config_path = "config/strawberries_config.json"
    if not os.path.exists(config_path):
        config_path = "../config/strawberries_config.json"

    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Config: {config}")

    width = config["width"]
    height = config["height"]

    # User's recipe expects Raw Pixel Coordinates (m, n)
    # m = 1..Width, n = 1..Height
    print("Creating raw pixel grid...")

    # Use float64 for coordinates as well
    x_indices = jnp.arange(1, width + 1, dtype=jnp.float64)
    y_indices = jnp.arange(1, height + 1, dtype=jnp.float64)

    xv, yv = jnp.meshgrid(x_indices, y_indices)
    # xv, yv shape: (Height, Width)

    print("Compiling pipeline...")
    # The pipeline works on full arrays, so we just JIT it.
    pipeline_jit = jax.jit(lambda x, y: strawberry_pipeline(x, y, config))

    print("Rendering...")
    # Pass 2D arrays directly
    image = pipeline_jit(xv, yv)

    # Image should be (H, W, 3)
    image = jnp.clip(image, 0.0, 1.0)

    output_path = config["output_file"]
    print(f"Saving to {output_path}...")
    plt.imsave(output_path, image)
    print("Done!")


if __name__ == "__main__":
    main()
