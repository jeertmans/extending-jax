import jax.numpy as jnp
from jax import Array
import chex

from rms_norm import rms_norm


def rms_norm_ref(x: Array, eps=1e-5) -> Array:
    scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return x / scale


def test_rms_norm() -> None:
    x = jnp.linspace(-0.5, 0.5, 15).reshape((3, 5))
    chex.assert_trees_all_close(rms_norm(x), rms_norm_ref(x), rtol=1e-5)
