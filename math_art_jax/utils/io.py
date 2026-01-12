import matplotlib.pyplot as plt
from jaxtyping import Float, Array
import numpy as np


def save_image(data: Float[Array, "H W"], filepath: str, cmap: str = "inferno", dpi: int = 300):
    """
    Save the JAX array as an image file.
    """
    # Convert to numpy for matplotlib
    img_np = np.array(data)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np, cmap=cmap, origin="lower")
    plt.axis("off")
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()
