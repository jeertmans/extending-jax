def test_numpy_wrapper_matches_reference():
    import jax.numpy as jnp
    import numpy as np
    from rms_norm import rms_norm_numpy

    def rms_norm_ref(x, eps=1e-5):
        scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
        return x / scale

    x = np.linspace(-0.5, 0.5, 10).astype(np.float32)
    y = rms_norm_numpy(x)
    y_ref = np.asarray(rms_norm_ref(x))

    np.testing.assert_allclose(y, y_ref, rtol=1e-5)


def test_numpy_wrapper_shape():
    import numpy as np
    from rms_norm import rms_norm_numpy

    x = np.arange(20, dtype=np.float32)
    y = rms_norm_numpy(x)

    assert y.shape == x.shape
