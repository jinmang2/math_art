import argparse
import logging
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_execution_time(func):
    """함수 실행 시간을 측정하고 로그를 남기는 데코레이터"""

    def wrapper(*args, **kwargs):
        logging.debug(f"Executing {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(
            f"{func.__name__} executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


def safe_exp(x) -> np.ndarray:
    """Safe exponential function with clipping to prevent overflow."""
    return np.exp(np.clip(x, -1000, 700))


@log_execution_time
def _prepare_s_range(width: int, height: int, n_range: int) -> np.ndarray:
    return np.tile(np.arange(n_range).reshape(n_range, 1, 1), (1, height, width))


@log_execution_time
def _prepare_p_table(X: np.ndarray, Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.arctan(
        np.tan(
            2 * np.sin(5 * S) * X
            - 2 * np.cos(5 * S) * Y
            + 3 * np.cos(5 * S)
            + (3 * np.cos(14 * X - 19 * Y + 5 * S)) / 200
        )
    )


@log_execution_time
def _prepare_q_table(X: np.ndarray, Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.arctan(
        np.tan(
            2 * (np.cos(5 * S) * X + np.sin(5 * S) * Y + 2 * np.cos(4 * S))
            + (3 * np.cos(18 * X + 15 * Y + 4 * S)) / 200
        )
    )


@log_execution_time
def _prepare_w_table(
    X: np.ndarray, Y: np.ndarray, width: int, height: int, n_range: int
) -> np.ndarray:
    W_table = np.zeros((n_range, height, width))

    for s in range(1, n_range + 1):
        left = (
            np.cos(
                28**s * 25**-s * (np.cos(2 * s) * X + np.sin(2 * s) * Y)
                + 2 * np.sin(5 * s)
            )
            ** 2
        )
        right = (
            np.cos(
                28**s * 25**-s * (np.cos(2 * s) * Y - np.sin(2 * s) * X)
                + 2 * np.sin(6 * s)
            )
            ** 2
        )
        exp_exponent = -3 * (left * right + (-97 / 100))
        W_table[s - 1] = safe_exp(-safe_exp(exp_exponent))

    return W_table


@log_execution_time
def _prepare_a_table(v: int, S: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    s_const = -safe_exp(-1000 * (S - 1 / 2))
    weighted = -safe_exp(
        v
        * (
            (5 / 4) * (1 - P) * (Q**2)
            + (P) ** 2
            - 11 / 20
            + (np.arctan(100 * (v - 100)) / (10 * np.pi))
        )
    )
    penalty = -safe_exp(v * (Q**2 + P**2 - 1))
    return safe_exp(s_const + weighted + penalty)


@log_execution_time
def _prepare_u_table(
    X: np.ndarray,
    Y: np.ndarray,
    S: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    R0: np.ndarray,
    R1: np.ndarray,
) -> np.ndarray:
    left = -safe_exp(
        -100
        * (
            P
            - 57 / 100
            - (3 / 20 + np.cos(7 * Q + 2 * S) / 10)
            * np.cos(
                (10 + 3 * np.cos(14 * S)) * np.arccos(R0)
                + 3 / 10 * np.cos(45 * X + 47 * Y + np.cos(17 * X))
                + 2 * np.cos(5 * S)
            )
        )
    )
    right = -safe_exp(1000 * (P - 18 / 25 + (3 / 2 * Q) ** 8))
    M = safe_exp(left + right)

    left = -safe_exp(
        100
        * (
            P
            - 37 / 50
            - (3 / 20 + np.cos(8 * Q + 5 * S) / 10)
            * np.cos(
                (10 + 3 * np.cos(16 * S)) * np.arccos(R1)
                + 3 / 10 * np.cos(38 * X - 47 * Y + np.cos(19 * X))
                + 2 * np.cos(4 * S)
            )
        )
    )
    right = -safe_exp(-1000 * (P - 71 / 100 - (3 / 2 * Q) ** 8))
    N = safe_exp(left + right)

    return 1 - (1 - M) * (1 - N)


@log_execution_time
def _prepare_r_table(t: int, P: np.ndarray, Q: np.ndarray) -> np.ndarray:

    def E(t: int, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        left_term = 1000 / np.sqrt(20) * Q
        center_term = np.sqrt(5 * np.abs(20 - 20 * (1 - 2 * t) * P - 27 * t))
        right_term = (
            1 + 50 * np.sqrt(np.abs(4 * (200 - (20 * (1 - 2 * t) * P + 27 * t) ** 2)))
        ) ** -1

        return left_term * center_term * right_term

    E_t_s = E(t, P, Q)
    return E_t_s * safe_exp(-safe_exp(1000 * (np.abs(E_t_s) - 1)))


@log_execution_time
def _prepare_c_table(
    v: int,
    R: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    W_xy: np.ndarray,
) -> np.ndarray:
    cos_term = np.cos(10 * np.arccos(R)) * np.cos(25 / 2 * P)
    sin_term = np.sin(10 * np.arccos(R)) * np.sin(25 / 2 * P)

    left = (
        -safe_exp(v * (cos_term - 7 / 10 - W_xy / 5))
        - safe_exp(-v * (cos_term + 7 / 10 + W_xy / 5))
        - safe_exp(v * (sin_term - 7 / 10 - W_xy / 5))
        - safe_exp(-v * (sin_term + 7 / 10 + W_xy / 5))
    )
    right = -safe_exp(3 / 2 * ((Q) ** 2 + (P - 1 / 4) ** 2 - 21 / 50 + W_xy / 5))

    return safe_exp(left + right)


@log_execution_time
def _prepare_b_table(R: np.ndarray, P: np.ndarray) -> np.ndarray:
    return safe_exp(
        -safe_exp(-70 * (np.cos(20 * np.arccos(R)) * np.cos(25 * P) - 47 / 50))
    )


def F(x: np.ndarray) -> np.ndarray:
    """Compute the transformation function F.
    -> RGB 0~255로 mapping
    """
    e_val = safe_exp(-safe_exp(-1000 * x))
    x_val = np.abs(x) ** (safe_exp(-safe_exp(1000 * (x - 1))))
    return 255 * e_val * x_val


def L(v: int, Table: Dict[str, np.ndarray]) -> np.ndarray:
    """Computes L function using precomputed tables."""
    left_term_1 = np.cos(20 * np.arccos(Table["R0"])) * np.cos(25 * Table["P"])
    left_term_1 = 0.1 - (1 / 40) * left_term_1

    left_term_2 = 4 * v**2 - 13 * v + 11
    left_term_2 += np.cos(7 * Table["S"] + v * Table["S"])
    left_term_2 += 20 * safe_exp(
        -safe_exp(-70 * (Table["C20"] - 1 / 2))
    ) + 20 * safe_exp(-safe_exp(-10 * (Table["C10"] - 1 / 2)))

    return left_term_1 * left_term_2 * Table["A4"] * Table["A1000"] + Table["B"]


def H(v: int, Table: Dict[str, np.ndarray]) -> np.ndarray:
    """Main Function. 딸기를 loop을 돌면서 생성"""

    def f(r: int) -> np.ndarray:
        return (
            (1 - Table["A1000"][r])
            * (1 - safe_exp(-safe_exp(-1000 * (r - 0.5))) * Table["U"][r])
            * (1 - 1.25 * Table["A4"][r])
        )

    def g(v: int, s: int) -> np.ndarray:
        res = (
            (5 - 3 * (v - 1) ** 2 + Table["W_xy"])
            / 10
            * safe_exp(-safe_exp(-np.abs(71 / 10 - 10 * Table["P"][s])))
            * Table["U"][s]
        )
        res += Table["A1000"][s] * (1 - Table["U"][s]) * L_val[s]
        return res

    L_val = L(v, Table)
    f_cumprod = np.ones_like(Table["W_xy"])
    H_val = np.zeros_like(Table["W_xy"])
    for s in tqdm(range(1, 31)):
        f_cumprod *= f(s - 1)
        H_val += f_cumprod * g(v, s)

    return H_val


def create_meshgrid(
    width: int, height: int, denominator: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates a meshgrid for X and Y values."""
    x_vals = (np.arange(width) + 1 - width // 2) / (denominator or width // 2)
    y_vals = (height // 2 - np.arange(height)) / (denominator or height // 2)
    return np.meshgrid(x_vals, y_vals)


PREPARE_FUNCTIONS = {
    "S": _prepare_s_range,
    "P": _prepare_p_table,
    "Q": _prepare_q_table,
    "W": _prepare_w_table,
    "A": _prepare_a_table,
    "R": _prepare_r_table,
    "U": _prepare_u_table,
    "C": _prepare_c_table,
    "B": _prepare_b_table,
}


def prepare_tables(
    width: int, height: int, X: np.ndarray, Y: np.ndarray
) -> Dict[str, np.ndarray]:
    S = PREPARE_FUNCTIONS["S"](width, height, n_range=31)
    P = PREPARE_FUNCTIONS["P"](X, Y, S)
    Q = PREPARE_FUNCTIONS["Q"](X, Y, S)
    W = PREPARE_FUNCTIONS["W"](X, Y, width, height, n_range=40)
    W_xy = W.sum(axis=0)
    R0 = PREPARE_FUNCTIONS["R"](0, P, Q)
    R1 = PREPARE_FUNCTIONS["R"](1, P, Q)

    return {
        "S": S,
        "P": P,
        "Q": Q,
        "W_xy": W_xy,
        "A4": PREPARE_FUNCTIONS["A"](4, S, P, Q),
        "A1000": PREPARE_FUNCTIONS["A"](1000, S, P, Q),
        "R0": R0,
        "R1": R1,
        "U": PREPARE_FUNCTIONS["U"](X, Y, S, P, Q, R0, R1),
        "C10": PREPARE_FUNCTIONS["C"](10, R0, P, Q, W_xy),
        "C20": PREPARE_FUNCTIONS["C"](20, R0, P, Q, W_xy),
        "B": PREPARE_FUNCTIONS["B"](R0, P),
    }


def compute_RGB(Table: Dict[str, np.ndarray]) -> np.ndarray:
    """RGB 계산하여 image로 변환"""
    R = F(H(0, Table))
    G = F(H(1, Table))
    B = F(H(2, Table))

    img = np.stack([R, G, B], axis=-1).astype(np.uint8)
    return img


# Generate image with vectorized operations
def generate_image(
    width: int = 2000, height: int = 1200, denominator: int | None = None
) -> np.ndarray:
    """Generates the full image by computing necessary tables and applying transformations."""
    X, Y = create_meshgrid(width, height, denominator)
    Table = prepare_tables(width, height, X, Y)

    logging.info("Generating image using vectorized RGB computation...")
    image = compute_RGB(Table)
    return image


# Run the optimized function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="strawberries.png")
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--use_height", action="store_false")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    denominator = None
    if args.use_height:
        denominator = args.height // 2

    image = generate_image(
        width=args.width,
        height=args.height,
        denominator=denominator,
    )
    plt.imsave(args.filename, image)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
