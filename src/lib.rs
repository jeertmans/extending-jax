use pyo3::prelude::*;

fn xla_rms_norm<T: xla::NativeType>(
    builder: &xla::XlaBuilder,
    eps: T,
    x: &xla::XlaOp,
) -> xla::Result<xla::XlaOp> {
    let eps = builder.c0(eps)?.convert(x.ty()?)?;
    let norm_x = (x * x)?.reduce_mean(&[-1], true)?;
    let x_normalized = (x * (norm_x + eps)?.rsqrt()?)?;
    Ok(x_normalized)
}

#[pyfunction(name = "rms_norm")]
fn py_rms_norm<'py>(eps: f32, x: &'py xla::PjRtBuffer, y: &'py xla::PjRtBuffer) -> PyResult<()> {
    let builder = xla::XlaBuilder::new("rms_norm");
    let client = xla::PjRtClient::cpu()?;
    let input_shape = x.on_device_shape()?;
    let input_param = builder.parameter_s(0, &input_shape, "x")?;
    let op = xla_rms_norm(&builder, eps, &input_param)?;
    let comp = op.build()?;
    let exe = client.compile(&comp)?;
    let _result = exe.execute_b(&[x, y])?;

    Ok(())
}

#[pymodule]
fn _rms_norm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_rms_norm, m)?)?;
    Ok(())
}
