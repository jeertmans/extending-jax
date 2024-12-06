use std::ffi::CString;

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

#[pymodule]
fn _rms_norm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let name = CString::new("rms_norm").unwrap();
    let f: fn(*mut ffi::XLA_FFI_CallFrame) -> *mut ffi::XLA_FFI_Error =
        |call_frame| unsafe { ffi::RmsNorm(call_frame) };
    m.add("rms_norm", PyCapsule::new(m.py(), f, Some(name))?)?;
    Ok(())
}
