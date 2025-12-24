use pyo3::prelude::*;

pub mod py_many_bubbles;
pub mod py_many_bubbles_legacy;
pub mod py_two_bubbles;
pub mod py_utils;

#[pymodule]
#[pyo3(name = "bubble_gw")]
fn bubble_gw(py: Python, module_parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module_two_bubbles = PyModule::new(module_parent.py(), "two_bubbles")?;
    module_two_bubbles.add_class::<py_two_bubbles::PyGravitationalWaveCalculator>()?;

    let module_many_bubbles_legacy = PyModule::new(module_parent.py(), "many_bubbles_legacy")?;
    module_many_bubbles_legacy.add_class::<py_many_bubbles_legacy::PyBulkFlow>()?;
    module_many_bubbles_legacy.add_class::<py_many_bubbles_legacy::PyLattice>()?;
    module_many_bubbles_legacy.add_class::<py_many_bubbles_legacy::PyPoissonNucleation>()?;
    module_many_bubbles_legacy.add_class::<py_many_bubbles_legacy::PyManualNucleation>()?;
    module_many_bubbles_legacy.add_class::<py_many_bubbles_legacy::PyBubbleFormationSimulator>()?;
    module_many_bubbles_legacy.add_function(wrap_pyfunction!(
        py_many_bubbles_legacy::py_generate_bubbles_exterior,
        module_parent
    )?)?;

    let module_many_bubbles = PyModule::new(module_parent.py(), "many_bubbles")?;
    module_many_bubbles.add_class::<py_many_bubbles::PyBulkFlow>()?;
    module_many_bubbles.add_class::<py_many_bubbles::PyLatticeBubbles>()?;
    module_many_bubbles.add_class::<py_many_bubbles::PyBuiltInLattice>()?;
    module_many_bubbles.add_class::<py_many_bubbles::PyIsometry3>()?;

    let module_utils = PyModule::new(module_parent.py(), "utils")?;
    module_utils.add_function(wrap_pyfunction!(py_utils::sample, module_parent)?)?;
    module_utils.add_function(wrap_pyfunction!(py_utils::sample_arr, module_parent)?)?;

    module_parent.add_submodule(&module_two_bubbles)?;
    module_parent.add_submodule(&module_many_bubbles_legacy)?;
    module_parent.add_submodule(&module_many_bubbles)?;
    module_parent.add_submodule(&module_utils)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bubble_gw.two_bubbles", module_two_bubbles)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bubble_gw.many_bubbles_legacy", module_many_bubbles_legacy)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bubble_gw.many_bubbles", module_many_bubbles)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("bubble_gw.utils", module_utils)?;
    Ok(())
}
