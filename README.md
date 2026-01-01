# Extending JAX with Rust code

Experiments on porting the C++ *Extending JAX* tutorial to Rust.

## Related Links

- [dfm/extending-jax](https://github.com/dfm/extending-jax): Extending JAX with custom C++ and CUDA code
- JAX: [Foreign function interface (FFI)](https://docs.jax.dev/en/latest/ffi.html)
- [End-to-end example usage for JAX's foreign function interface](https://github.com/jax-ml/jax/tree/main/examples/ffi)

## Installation

To build this project from source, you need Rust and a modern C/C++ compiler:

```bash
git clone https://github.com/jeertmans/extending-jax.git
cd extending-jax
pip install .
```

## Development

If you want to develop or test the project, I recommend using [uv](https://uvproject.io/) to manage the virtual environment.

First, make sure to install all dependencies[^1]:

```bash
uv sync  # To install project locally (it builds and links automatically Rust bindings)
```

[^1]: Actually, you need to run this command to install JAX, because it is a build dependency for the Rust code. Otherwise, commands like `cargo check` will fail.

Then, you can use `cargo check` to check that the Rust code compiles, and `uv sync --force-reinstall` to force rebuilding the Rust / C++ files after making changes.

You can also run tests with `uv run pytest` or rapidly test the main function with a one-liner.

```bash
uv run python -c "from rms_norm import rms_norm as f;import jax.numpy as jnp;print(f(jnp.ones(4)))"
```

## Future Work

Currently, the following features are implemented:

- [x] Rust implementation of the `rms_norm` function
- [x] Basic testing of the `rms_norm` function

Here is the list of things that could be added in the future:

- [ ] Add forward and backward implementation in Rust
- [ ] Add GPU implementation (CUDA support)
- [ ] Add more tests and benchmarks
- [ ] Document how it works and why we still need some (basic) C++ code to use the JAX FFI

## Contributing

Unfortunately, my available time for this experiment is limited. If you want to contribute, **please** feel free to open issues or pull requests!

Related discussions:
- [JAX discussion on extending with Rust](https://github.com/jax-ml/jax/discussions/24187)
- [PyO3 discussion on exporting function pointers](https://github.com/PyO3/pyo3/discussions/4772)
