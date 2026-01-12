# Math Art JAX

High-performance, memory-efficient Mathematical Art Rendering Engine using JAX.
Inspired by the mathematical art of Hamid Naderi Yeganeh.

## Overview

This engine is designed to handle high-density layering (up to 150+ layers) and extreme precision calculations using JAX's functional programming paradigm. It leverages `vmap` for parallel pixel processing and `lax.scan` for efficient memory usage during layer accumulation.

## Installation

Requires Python 3.9+ and JAX.

```bash
pip install jax jaxlib jaxtyping matplotlib numpy
```

_(Note: Install the appropriate `jaxlib` version for your hardware, especially if using GPU/TPU)_

## Structure

```text
math_art_jax/
├── core/                   # Atomic mathematical operations
│   ├── primitives.py       # F(x), soft_mask, high-power filters
│   ├── logic.py            # Soft-Boolean logic
│   └── noise.py            # Procedural noise generation
├── warpers/                # Coordinate system distortion functions
│   ├── basic.py            # Affine transformations
│   ├── polar.py            # Polar/Spiral warps
│   ├── perspective.py      # Depth simulation
│   └── organic.py          # Non-linear organic distortions
├── generators/             # Art generation recipes
│   ├── implicit_shader.py  # Implicit function renderers
│   ├── parametric_shape.py # Parametric curve renderers
│   └── clustering.py       # Particle/Cluster renderers
├── engine/                 # JAX execution pipeline
│   ├── renderer.py         # Main rendering orchestration
│   └── accumulator.py      # Layer synthesis
└── utils/                  # IO and helpers
```

## Usage

(Coming soon)
