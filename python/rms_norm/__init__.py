# This file below is a modified version of: https://github.com/jax-ml/jax/blob/main/examples/ffi/src/jax_ffi_example/rms_norm.py.
#
# Original license header starts now.
#
# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np

import rms_norm._rms_norm as rms_norm_lib

jax.ffi.register_ffi_target("rms_norm", rms_norm_lib.rms_norm(), platform="cpu")


def rms_norm(x, eps=1e-5):
    # We only implemented the `float32` version of this function, so we start by
    # checking the dtype. This check isn't strictly necessary because type
    # checking is also performed by the FFI when decoding input and output
    # buffers, but it can be useful to check types in Python to raise more
    # informative errors.
    if x.dtype != jnp.float32:
        raise ValueError("Only the float32 dtype is implemented by rms_norm")

    call = jax.ffi.ffi_call(
        # The target name must be the same string as we used to register the target
        # above in `register_custom_call_target`
        "rms_norm",
        # In this case, the output of our FFI function is just a single array with
        # the same shape and dtype as the input. We discuss a case with a more
        # interesting output type below.
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        # The `vmap_method` parameter controls this function's behavior under `vmap`
        # as discussed below.
        vmap_method="broadcast_all",
    )

    # Note that here we're use `numpy` (not `jax.numpy`) to specify a dtype for
    # the attribute `eps`. Our FFI function expects this to have the C++ `float`
    # type (which corresponds to numpy's `float32` type), and it must be a
    # static parameter (i.e. not a JAX array).
    return call(x, eps=np.float32(eps))


if hasattr(rms_norm_lib, "rms_norm_numpy"):

    def rms_norm_numpy(x, eps=1e-5):
        x = np.asarray(x).astype(np.float32)
        y = rms_norm_lib.rms_norm_numpy(x, eps)
        return jnp.asarray(y)
