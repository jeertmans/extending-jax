# Extending JAX with Rust code

Experiments on porting the C++ *Extending JAX* tutorial to Rust.


## Building

To build this project, you need Rust and uv (or any Python project builder).

```bash
cargo check  # To check that it compiles (you might need to activate local venv)
uv sync  # To install project locally (it builds and links automatically Rust bindings)
uv sync --force-reinstall  # To force rebuilding Rust / C++ files
uv run pytest  # To run tests
uv run python -c "from rms_norm import rms_norm as f;import jax.numpy as jnp;print(f(jnp.ones(4)))"  # To test function
```