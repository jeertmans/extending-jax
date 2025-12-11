# Extending JAX with Rust code

Experiments on porting the C++ *Extending JAX* tutorial to Rust.

## Related Links

- [dfm/extending-jax](https://github.com/dfm/extending-jax): Extending JAX with custom C++ and CUDA code
- JAX: [Foreign function interface (FFI)](https://docs.jax.dev/en/latest/ffi.html)
- [End-to-end example usage for JAX's foreign function interface](https://github.com/jax-ml/jax/tree/main/examples/ffi)

## Building

To build this project, you need Rust and uv (or any Python project manager).

```bash
uv sync  # To install project locally (it builds and links automatically Rust bindings)
cargo check  # To check that it compiles (you need to activate local venv and have JAX installed)
uv sync --force-reinstall  # To force rebuilding Rust / C++ files
uv run pytest  # To run tests
uv run python -c "from rms_norm import rms_norm as f;import jax.numpy as jnp;print(f(jnp.ones(4)))"  # To test function
```

## Contributing

Unfortunately, my available time for this experiment is limited. If you want to contribute, **please** feel free to open issues or pull requests!

Looks at https://github.com/jax-ml/jax/discussions/24187 and https://github.com/PyO3/pyo3/discussions/4772 for more context about the challenges faced here.
