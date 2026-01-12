import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Callable, Tuple, Any, NamedTuple


# State Definition (Memory Efficient Carry)
class AccumulatorState(NamedTuple):
    transmission: Float[Array, "H W"]  # Cumulative transparency (1.0 -> 0.0)
    color_buffer: Float[Array, "H W 3"]  # Accumulated color (RGB)


def recursive_blend(
    shader_fn: Callable[[int, Any], Tuple[Float[Array, "H W"], Float[Array, "H W 3"]]],
    init_state: AccumulatorState,
    layer_indices: Float[Array, "L"],
    params: Any,
) -> AccumulatorState:
    """
    Executes the recursive blending loop using jax.lax.scan.

    Args:
        shader_fn: A function that takes (layer_index, params) and returns:
                   - mask_alpha (H, W): Opacity of the current layer (0.0 to 1.0)
                   - layer_color (H, W, 3): RGB color of the current layer
        init_state: Initial transmission (ones) and color buffer (zeros).
        layer_indices: Array of layer indices to iterate over.
        params: Static parameters required for the shader.

    Returns:
        Final AccumulatorState.
    """

    def scan_body(carry: AccumulatorState, index: int) -> Tuple[AccumulatorState, None]:
        curr_trans, curr_color = carry

        # Shader returns (alpha_trans, color_contribution)
        # alpha_trans (H, W): Amount to block the background (for T update).
        # color_contribution (H, W, 3): The RGB value designated to be added to the canvas at this layer.
        #                               (Pre-weighted by its own internal alpha/mask logic if needed).
        alpha_trans, color_contrib = shader_fn(index, params)

        # Contribution is scaled by current transmission
        term = curr_trans[:, :, None] * color_contrib

        new_color = curr_color + term
        new_trans = curr_trans * (1.0 - alpha_trans)

        new_state = AccumulatorState(transmission=new_trans, color_buffer=new_color)
        return new_state, None

    final_state, _ = jax.lax.scan(scan_body, init_state, layer_indices)
    return final_state
