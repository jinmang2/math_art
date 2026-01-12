import jax
import jax.numpy as jnp
from typing import Dict, Any, NamedTuple
from jaxtyping import Float, Array

from math_art_jax.layers import shaders as sh
from math_art_jax.warpers import organic as warp
from math_art_jax.engine.accumulator import recursive_blend, AccumulatorState
from math_art_jax.core.operators import safe_exp


class StrawberryParams(NamedTuple):
    w_range: int = 40
    s_range: int = 40
    width: int = 2000
    height: int = 1200


def strawberry_pipeline(
    x: Float[Array, "H W"], y: Float[Array, "H W"], config: Dict[str, Any]
) -> Float[Array, "H W 3"]:
    """
    The orchestrator recipe for the Strawberry artwork.
    """
    # 1. Coordinate Scaling
    # Map [1, 2000] to approx [-1.66, 1.66]
    x_scaled = (x - 1000.0) / 600.0
    y_scaled = (601.0 - y) / 600.0

    # 2. Configuration
    w_limit = int(config.get("w_sum_range", 40))
    s_limit = int(config.get("s_range", 40))

    # 2b. Pre-computation: W_xy texture map
    w_indices = jnp.arange(1, w_limit + 1, dtype=jnp.float64)

    def w_scan_body(carry, s):
        w_val = sh.strawberry_w_texture(x_scaled, y_scaled, s)
        return carry + w_val, None

    w_xy_map, _ = jax.lax.scan(w_scan_body, jnp.zeros_like(x_scaled, dtype=jnp.float64), w_indices)

    # 3. Layer Shader Definition
    ShaderContext = NamedTuple(
        "ShaderContext", [("x", Float[Array, "H W"]), ("y", Float[Array, "H W"]), ("w_map", Float[Array, "H W"])]
    )
    CONTEXT = ShaderContext(x_scaled, y_scaled, w_xy_map)

    def layer_shader_fn(s_idx: int, params: ShaderContext):
        s = s_idx.astype(jnp.float64)

        # A. Warp Coordinates
        P = warp.strawberry_p_warp(params.x, params.y, s)
        Q = warp.strawberry_q_warp(params.x, params.y, s)

        # B. Compute Auxiliaries
        R0 = sh.strawberry_seed_r(0.0, P, Q)
        R1 = sh.strawberry_seed_r(1.0, P, Q)

        M = sh.strawberry_m_func(params.x, params.y, s, P, Q, R0)
        N = sh.strawberry_n_func(params.x, params.y, s, P, Q, R1)
        U = sh.strawberry_u_func(M, N)

        A4 = sh.strawberry_occlusion_a(4.0, s, P, Q)
        A1000 = sh.strawberry_occlusion_a(1000.0, s, P, Q)

        # C. Compute RGB Components (L vector)
        B_val = sh.strawberry_b_func(R0, P)
        C10 = sh.strawberry_c_func(10.0, R0, P, Q, params.w_map)
        C20 = sh.strawberry_c_func(20.0, R0, P, Q, params.w_map)

        L0 = sh.strawberry_color_l(0.0, s, P, Q, params.w_map, A4, A1000, R0, C10, C20, B_val)
        L1 = sh.strawberry_color_l(1.0, s, P, Q, params.w_map, A4, A1000, R0, C10, C20, B_val)
        L2 = sh.strawberry_color_l(2.0, s, P, Q, params.w_map, A4, A1000, R0, C10, C20, B_val)
        L_vec = jnp.stack([L0, L1, L2], axis=-1)

        # D. Color Modulation G (The Color Term)
        exp_g = safe_exp(-safe_exp(-jnp.abs(7.1 - 10.0 * P)))

        g0 = (2.0 + params.w_map) / 10.0
        g1 = (5.0 + params.w_map) / 10.0
        g2 = (2.0 + params.w_map) / 10.0
        factor_g = jnp.stack([g0, g1, g2], axis=-1)

        U_exp = jnp.expand_dims(U, axis=-1)
        P_exp_g = jnp.expand_dims(exp_g, axis=-1)
        A1000_exp = jnp.expand_dims(A1000, axis=-1)

        # g_vec is the ACTUAL COLOR of this layer
        g_vec = factor_g * P_exp_g * U_exp + A1000_exp * (1.0 - U_exp) * L_vec

        # E. Transmission/Occlusion Logic (The f(r) Term)
        # f(r) = (1 - A1000)(1 - exp(...)U)(1 - 1.25 A4)
        # If f(r) = 1 (Transparent), then alpha = 0.
        # If f(r) = 0 (Opaque), then alpha = 1.

        mask_exp_u = safe_exp(-safe_exp(-1000.0 * (s - 0.5))) * U
        f_val = (1.0 - A1000) * (1.0 - mask_exp_u) * (1.0 - 1.25 * A4)

        # Accumulator expects 'alpha' such that new_T = old_T * (1 - alpha).
        # We want new_T = old_T * f_val.
        # So 1 - alpha = f_val => alpha = 1.0 - f_val.

        alpha_trans = 1.0 - f_val

        # We perform NO masking on g_vec here.
        # The Accumulator will multiply g_vec by Current Transmission (T).
        # If this layer is opaque (alpha=1, f=0), it draws fully (scaled by previous T),
        # and then reduces NEXT T to 0.

        return alpha_trans, g_vec

    # 4. Execution
    # Front-to-Back (1 -> 40).
    # s=1 (Front) draws first with T=1. Masks s=2 via alpha_trans.
    s_indices = jnp.arange(1, s_limit + 1, dtype=jnp.float64)

    init_state = AccumulatorState(
        transmission=jnp.ones_like(x_scaled, dtype=jnp.float64),
        color_buffer=jnp.zeros((x.shape[0], x.shape[1], 3), dtype=jnp.float64),
    )

    final_state = recursive_blend(layer_shader_fn, init_state, s_indices, CONTEXT)

    # 5. Final Tone Mapping
    return sh.strawberry_final_f(jnp.abs(final_state.color_buffer)) / 255.0
