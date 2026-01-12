---
trigger: model_decision
description: 이전 딸기 수식을 구현하면서 실수했던 부분을 교정한 knowledge 정리
---

# Principles of Math-to-Code Translation

## 1. Explicit Domain Mapping (The Canvas Contract)

Mathematical formulas operate in abstract, continuous domains (e.g., $x \in [-1, 1]$), whereas software engines operate in discrete, implementation-specific domains (e.g., `uint32` pixel indices).
**Principle:** Never implicitly trust raw inputs. Establish a strict **Coordinate Transformation Layer** at the system's entry point. This layer acts as the "Contract" that normalizes system inputs (Indices) into the formula's expected Mathematical Domain (Reals) before any logic is executed.

- _Failure Mode:_ Passing array indices `0..2000` into `exp(x)` causes immediate overflow/saturation, producing binary (All-Black/White) outputs.

## 2. Deciphering the Generative Logic (The Execution Arrow)

Mathematical notation (e.g., $\prod, \sum$) is static and often commutative, but computational rendering is temporal and directional. The order of operations implies a specific physical model (e.g., Painter's Algorithm vs. Ray Accumulation).
**Principle:** Do not just translate operators; translate the **Generative Process**. Analyze whether the formula implies a state evolution (e.g., $S_{new} = f(S_{old})$) or independent superposition.

- _Check:_ Is the product term representing "Transmission decay" (Front-to-Back) or "Masking" (Back-to-Front)?
- _Failure Mode:_ Reversing the loop order in a transmission-based model creates "X-Ray" or inverted occlusion effects.

## 3. Semantic Variable Separation (The "Type System" of Formulas)

In compact mathematical notation, a single term like $(1 - U)$ often physically conflates a "Control Signal" (Visibility) with a "Data Signal" (Color Intensity).
**Principle:** In code, forcibly separate **Control Flow** (Alpha, Masks, Thresholds) from **Data Flow** (RGB Coloring, Displacement vectors).

- _Practice:_ Even if the formula says $Color = (1-Mask) \times C$, implement the engine to return `{ alpha: Mask, color: C }` and handle the mixing in a dedicated blending stage.
- _Failure Mode:_ Conflating "Zero Visibility" with "Black Color" makes invisible objects cast "Dark Shadows" instead of allowing the background to pass through.
