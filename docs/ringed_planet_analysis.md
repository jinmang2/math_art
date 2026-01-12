# Ringed Planet Disrupted by a Rocky Planet - Analysis & Plan

## 1. Domain & Coordinates

Input Image size: $2000 \times 1150$.
Coordinate Norm:

- $x' = (m - 1170) / 1000$ (where $m = 1 \dots 2000$)
- $y' = (576 - n) / 1000$ (where $n = 1 \dots 1150$)
  This maps the domain to roughly $[-1.1, 0.8]$ in X and $[-0.5, 0.5]$ in Y.

## 2. Structural Analysis

The artwork is composed of three primary accumulated structures and a background/composition logic.

### A. The Accumulators (Scanned Terms)

Unlike the Strawberry (Front-to-Back Recursive Blending), this artwork constructs **Global Maps** first.

1.  **Ring Structure ($Q$)**
    - Formula: $Q(x,y) = \prod_{s=1}^{40} ...$
    - Type: **Product Accumulation**. Represents the transparency/shape of the rings.
2.  **Debris/Cloud ($A$)**
    - Formula: $A(x,y) = \sum_{s=1}^{40} ...$
    - Type: **Summation Accumulation**. Represents the density of the ejected material.
3.  **Planet Noise ($E$)**
    - Formula: $E(x,y) = \sum_{s=1}^{50} ...$
    - Type: **Summation Accumulation**. Fractal noise for the rocky planet's texture.

### B. The Atomic Functions

- **Tone Mapping ($F$)**: $|255 e^{-1000x} |x|^{...}|$. Identical structure to Strawberry.
- **Noise Basis ($N_s$)**: Cos/Sin fractal basis with frequency $6^s, 8^s, 9^s$. High frequency.
- **Planet Shape ($R_v$)**: Defines the main sphere surface and lighting.
- **Debris Color ($L$)**: Color modulation for the debris.
- **Ring Texture ($K$)**: Texture details for the rings.

### C. Final Composition ($H_v$)

The final color channel $v$ (Red, Green, Blue) is a complex mixture:
$H_v = \text{Planet}_v + \text{Ring}_v \times \text{Shadow}_v + \text{Debris}_v$
Specifically:

- $R_v, P$: Rocky Planet Surface.
- $Q \cdot B_0$: Ring occlusion.
- $A \cdot (1-C_0) ...$: Debris contribution.

## 3. Implementation Strategy (The Pipeline)

### Phase 1: Engine Extension (If needed)

The current `accumulator.py` is specialized for `Transmission/Color` blending.
This artwork requires **Generic Map Generation**.

- We can use raw `jax.lax.scan` in the recipe directly, similar to how we computed `W_map` in Strawberries.
- **Constraint**: `s` ranges differ ($40$ vs $50$).
  - Strategy: Run a single scan up to $s=50$.
  - Inside scan body: If $s > 40$, terms for $Q, A$ contribute Identity ($1$ for prod, $0$ for sum).

### Phase 2: Atomic Shaders (`src/math_art_jax/layers/planet_shaders.py`)

Implement the alphabet soup:

- `planet_q_term(s, ...)`
- `planet_a_term(s, ...)`
- `planet_e_term(s, ...)`
- `planet_n_noise(s, ...)`
- `planet_final_h(...)`

### Phase 3: Recipe (`src/recipes/ringed_planet.py`)

1.  **Coordinate Transform**: Apply linear mapping.
2.  **Scan Loop ($s=1 \dots 50$)**:
    - Carry: `(prod_Q, sum_A, sum_E)`
    - Update logic detailed above.
3.  **Composition**:
    - Compute `R_v` using `E`.
    - Compute `H_v` combining `R, Q, A`.
4.  **Tone Mapping**: Apply `F`.

## 4. Complexity & Risks

- **Precision**: High exponents ($6^{50}$) in noise $N_s$. **Must use `float64`**.
- **Performance**: $N_s$ calls many trig functions per layer. 50 layers x 2M pixels.
  - $10^8$ complex evaluations. GPU recommended but CPU (vectorized) should handle it in minutes.
- **Typos**: The formula image is dense. Careful transcription required.

## 5. Directory Structure

```
src/
  math_art_jax/
    layers/
      planet_shaders.py  <-- NEW
  recipes/
    ringed_planet.py     <-- NEW
scripts/
  generate_ringed_planet.py <-- NEW
```
