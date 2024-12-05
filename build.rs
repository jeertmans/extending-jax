use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::str::from_utf8;

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    let output = Command::new(
        env::var("PYTHON")
            .ok()
            .or_else(|| pyo3_build_config::get().executable.clone())
            .unwrap_or_else(|| "python3".to_owned()),
    )
    .arg("-c")
    .arg("from jax.extend import ffi;print(ffi.include_dir())")
    .output()
    .expect("failed to execute process");

    let stdout = from_utf8(&output.stdout).unwrap();

    if !output.status.success() {
        let stderr = from_utf8(&output.stderr).unwrap();
        eprint!("{stdout}{stderr}");
        panic!(
            "could not retrieve xla include dir from 'jax.extend': {}",
            output.status
        );
    }

    let include_dir = stdout.trim();

    let path = PathBuf::from(include_dir);

    cxx_build::bridge("src/lib.rs")
        .file("src/ffi.cc")
        .std("c++17")
        .include(path)
        .compile("extending-jax");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/ffi.cc");
    println!("cargo:rerun-if-changed=include/ffi.h");
    println!("cargo:rerun-if-changed={include_dir}");
}
