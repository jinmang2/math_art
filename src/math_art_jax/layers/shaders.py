import jax.numpy as jnp
from jaxtyping import Float, Array
from math_art_jax.core.operators import safe_exp

# --- Strawberry Component Shaders ---


def strawberry_w_texture(x: Float[Array, "..."], y: Float[Array, "..."], s: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Computes W(x,y,s): The high-frequency surface texture.
    """
    factor = (28.0 / 25.0) ** s
    left_arg = factor * (jnp.cos(2 * s) * x + jnp.sin(2 * s) * y) + 2 * jnp.sin(5 * s)
    left = jnp.cos(left_arg) ** 2
    right_arg = factor * (jnp.cos(2 * s) * y - jnp.sin(2 * s) * x) + 2 * jnp.sin(6 * s)
    right = jnp.cos(right_arg) ** 2
    exp_exponent = -3.0 * (left * right - 0.97)
    return safe_exp(-safe_exp(exp_exponent))


def strawberry_occlusion_a(
    v: float, s: Float[Array, "..."], P: Float[Array, "..."], Q: Float[Array, "..."]
) -> Float[Array, "..."]:
    """
    Computes A(v, s, P, Q): The primary shape/occlusion masking term.
    """
    s_const = -safe_exp(-1000.0 * (s - 0.5))
    term1 = 1.25 * (1.0 - P) * (Q**2)
    term2 = P**2
    term3 = -0.55
    term4 = jnp.arctan(100.0 * (v - 100.0)) / (10.0 * jnp.pi)
    weighted = -safe_exp(v * (term1 + term2 + term3 + term4))
    penalty = -safe_exp(v * (Q**2 + P**2 - 1.0))
    return safe_exp(s_const + weighted + penalty)


def strawberry_seed_r(t: float, P: Float[Array, "..."], Q: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Computes R(t, P, Q): Helper for seed/dot placement.
    """
    left_term = (1000.0 / jnp.sqrt(20.0)) * Q
    abs_center = 5.0 * jnp.abs(20.0 - 20.0 * (1.0 - 2.0 * t) * P - 27.0 * t)
    center = jnp.sqrt(abs_center)

    inner_right = 4.0 * (200.0 - (20.0 * (1.0 - 2.0 * t) * P + 27.0 * t) ** 2)
    denom = 1.0 + 50.0 * jnp.sqrt(jnp.abs(inner_right))
    right = 1.0 / denom

    E_val = left_term * center * right
    return E_val * safe_exp(-safe_exp(1000.0 * (jnp.abs(E_val) - 1.0)))


def strawberry_b_func(R: Float[Array, "..."], P: Float[Array, "..."]) -> Float[Array, "..."]:
    inner = jnp.cos(20.0 * jnp.arccos(R)) * jnp.cos(25.0 * P) - 0.94
    return safe_exp(-safe_exp(-70.0 * inner))


def strawberry_c_func(
    v: float, R: Float[Array, "..."], P: Float[Array, "..."], Q: Float[Array, "..."], W_xy: Float[Array, "..."]
) -> Float[Array, "..."]:
    cos_val = jnp.cos(10.0 * jnp.arccos(R))
    sin_val = jnp.sin(10.0 * jnp.arccos(R))
    cos_term = cos_val * jnp.cos(12.5 * P)
    sin_term = sin_val * jnp.sin(12.5 * P)
    left = (
        -safe_exp(v * (cos_term - 0.7 - W_xy / 5.0))
        - safe_exp(-v * (cos_term + 0.7 + W_xy / 5.0))
        - safe_exp(v * (sin_term - 0.7 - W_xy / 5.0))
        - safe_exp(-v * (sin_term + 0.7 + W_xy / 5.0))
    )
    right = -safe_exp(1.5 * (Q**2 + (P - 0.25) ** 2 - 0.42 + W_xy / 5.0))
    return safe_exp(left + right)


def strawberry_m_func(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    s: Float[Array, "..."],
    P: Float[Array, "..."],
    Q: Float[Array, "..."],
    R0: Float[Array, "..."],
) -> Float[Array, "..."]:
    factor = 0.15 + jnp.cos(7.0 * Q + 2.0 * s) / 10.0
    arg_cos = (
        (10.0 + 3.0 * jnp.cos(14.0 * s)) * jnp.arccos(R0)
        + 0.3 * jnp.cos(45.0 * x + 47.0 * y + jnp.cos(17.0 * x))
        + 2.0 * jnp.cos(5.0 * s)
    )
    term_main = P - 0.57 - factor * jnp.cos(arg_cos)
    left = -safe_exp(-100.0 * term_main)
    right = -safe_exp(1000.0 * (P - 0.72 + (1.5 * Q) ** 8))
    return safe_exp(left + right)


def strawberry_n_func(
    x: Float[Array, "..."],
    y: Float[Array, "..."],
    s: Float[Array, "..."],
    P: Float[Array, "..."],
    Q: Float[Array, "..."],
    R1: Float[Array, "..."],
) -> Float[Array, "..."]:
    factor = 0.15 + jnp.cos(8.0 * Q + 5.0 * s) / 10.0
    arg_cos = (
        (10.0 + 3.0 * jnp.cos(16.0 * s)) * jnp.arccos(R1)
        + 0.3 * jnp.cos(38.0 * x - 47.0 * y + jnp.cos(19.0 * x))
        + 2.0 * jnp.cos(4.0 * s)
    )
    term_main = P - 0.74 - factor * jnp.cos(arg_cos)
    left = -safe_exp(100.0 * term_main)
    right = -safe_exp(-1000.0 * (P - 0.71 - (1.5 * Q) ** 8))
    return safe_exp(left + right)


def strawberry_u_func(M: Float[Array, "..."], N: Float[Array, "..."]) -> Float[Array, "..."]:
    return 1.0 - (1.0 - M) * (1.0 - N)


def strawberry_color_l(
    v_channel: float,
    s: Float[Array, "..."],
    P: Float[Array, "..."],
    Q: Float[Array, "..."],
    W_xy: Float[Array, "..."],
    A4: Float[Array, "..."],
    A1000: Float[Array, "..."],
    R0: Float[Array, "..."],
    C10: Float[Array, "..."],
    C20: Float[Array, "..."],
    B: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Computes L_v,s(x,y): The raw color contribution for channel v.
    """
    term1_cos = jnp.cos(20.0 * jnp.arccos(R0)) * jnp.cos(25.0 * P)
    left_1 = 0.1 - 0.025 * term1_cos

    poly_v = 4.0 * (v_channel**2) - 13.0 * v_channel + 11.0
    cos_s = jnp.cos(7.0 * s + v_channel * s)
    exp_c20 = 20.0 * safe_exp(-safe_exp(-70.0 * (C20 - 0.5)))
    exp_c10 = 20.0 * safe_exp(-safe_exp(-10.0 * (C10 - 0.5)))

    left_2 = poly_v + cos_s + exp_c20 + exp_c10

    return left_1 * left_2 * A4 * A1000 + B


def strawberry_final_f(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Global tone mapping F(x).
    """
    eps = 1e-12
    e_val = safe_exp(-safe_exp(-1000.0 * x))
    exponent = safe_exp(-safe_exp(1000.0 * (x - 1.0)))
    val = jnp.power(jnp.abs(x) + eps, exponent)
    return 255.0 * e_val * val
