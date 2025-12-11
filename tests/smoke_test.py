import jax.numpy as jnp
from rms_norm import rms_norm


def test_simple():
    x = jnp.linspace(-0.5, 0.5, 15).reshape((3, 5))
    rms_norm(x)
