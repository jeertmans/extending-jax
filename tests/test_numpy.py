def test_numpy_wrapper_matches_reference():
    import numpy as np
    from rms_norm import rms_norm_numpy, rms_norm_ref

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
