use pyo3::prelude::*;

pub mod many_bubbles_bindings;
pub mod two_bubbles_bindings;
pub mod utils_bindings;

#[pymodule]
fn bubble_gw(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let two_bubbles_module = PyModule::new(parent_module.py(), "two_bubbles")?;
    two_bubbles_module.add_class::<two_bubbles_bindings::PyGravitationalWaveCalculator>()?;

    let many_bubbles_module = PyModule::new(parent_module.py(), "many_bubbles")?;
    many_bubbles_module.add_class::<many_bubbles_bindings::PyBulkFlow>()?;
    many_bubbles_module.add_class::<many_bubbles_bindings::PyLattice>()?;
    many_bubbles_module.add_class::<many_bubbles_bindings::PyPoissonNucleation>()?;
    many_bubbles_module.add_class::<many_bubbles_bindings::PyManualNucleation>()?;
    many_bubbles_module.add_class::<many_bubbles_bindings::PyBubbleFormationSimulator>()?;

    let utils_module = PyModule::new(parent_module.py(), "utils")?;
    utils_module.add_function(wrap_pyfunction!(utils_bindings::sample, parent_module)?)?;
    utils_module.add_function(wrap_pyfunction!(utils_bindings::sample_arr, parent_module)?)?;

    parent_module.add_submodule(&two_bubbles_module)?;
    parent_module.add_submodule(&many_bubbles_module)?;
    parent_module.add_submodule(&utils_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bubble_gw.two_bubbles", two_bubbles_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bubble_gw.many_bubbles", many_bubbles_module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bubble_gw.utils", utils_module)?;
    Ok(())
}
