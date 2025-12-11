use std::ffi::c_void;

#[cfg(feature = "numpy")]
use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyCapsule};

fn rms_norm(eps: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len(), "x and y must have the same length");
    let mut sm = 0f32;
    let size = x.len();
    for xi in x {
        sm += xi * xi;
    }
    let scale = (sm / (size as f32) + eps).sqrt().recip();

    for i in 0..size {
        y[i] = x[i] * scale;
    }
}

#[cfg(feature = "numpy")]
#[pyfunction(name = "rms_norm_numpy")]
fn rms_norm_numpy<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    eps: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let mut y = Vec::with_capacity(x.len());
    rms_norm(eps, x.as_slice()?, y.as_mut_slice());
    Ok(PyArray1::from_vec(py, y))
}

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        // Expose to C++ our Rust function
        fn rms_norm(eps: f32, x: &[f32], y: &mut [f32]) -> ();
    }

    unsafe extern "C++" {
        include!("rms-norm/include/ffi.h");

        type XLA_FFI_Error;
        type XLA_FFI_CallFrame;

        // This is the C++ XLA compatible wrapper around our 'rms_norm' Rust function
        unsafe fn RmsNorm(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error;
    }
}

#[pyfunction(name = "rms_norm")]
fn rms_norm_jax(py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
    unsafe { PyCapsule::new_pointer(py, ffi::RmsNorm as *mut c_void, None) }
}

#[pymodule]
fn _rms_norm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rms_norm_jax, m)?)?;

    // This is just to compare the JAX version with the Rust-NumPy version
    #[cfg(feature = "numpy")]
    m.add_function(wrap_pyfunction!(rms_norm_numpy, m)?)?;

    Ok(())
}
