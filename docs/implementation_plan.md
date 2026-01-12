# Math Art JAX: Architectural Guidelines

## 1. Core Principle: Separation of Engine and Content

The library `math_art_jax` MUST provide only generic computational tools. It MUST NOT contain any artwork-specific constants or formulas.
All artwork logic MUST reside in the `recipes/` directory.

## 2. Directory Structure

```text
math_art/
├── math_art_jax/           # The Generic Engine
│   ├── core/               # Primitives (safe_exp, grid, logic)
│   ├── warpers/            # Coordinate transformations
│   └── engine/             # Generic Render Pipelines (scan_renderer, vmap_renderer)
├── recipes/                # The Art Content (User Space)
│   ├── strawberries.py     # Strawberry specific logic
│   ├── waterfall.py        # (Planned) Waterfall logic
│   └── bird_flight.py      # (Planned) Bird logic
└── scripts/
    └── generate_art.py     # Generic Runner (TODO: Unify individual scripts)
```

## 3. Recipe Interface

Each recipe should expose a `shader` function or a `Recipe` class that:

1.  Takes `(x, y)` coordinates and `config` dictionary.
2.  Returns `RGB` float definition.
3.  Utilizes `math_art_jax` primitives.

## 4. Next Steps for Implementation Agent

1.  **Analyze Formulas**: Read the provided formulas for Waterfall, Bird, etc.
2.  **Create Recipe**: Implement `recipes/<artwork>.py`.
3.  **Run**: Use the generic runner or create a minimal script.
