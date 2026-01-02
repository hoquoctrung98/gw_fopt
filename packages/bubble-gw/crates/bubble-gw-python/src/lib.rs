use pyo3::prelude::*;

pub mod py_many_bubbles;
pub mod py_many_bubbles_legacy;
pub mod py_two_bubbles;
pub mod py_utils;

#[pymodule]
#[pyo3(name = "bubble_gw")]
fn bubble_gw(py: Python, module_parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module_two_bubbles = PyModule::new(module_parent.py(), "two_bubbles")?;
    module_two_bubbles.add_class::<py_two_bubbles::py_gw_calc::PyGravitationalWaveCalculator>()?;

    let module_many_bubbles_legacy = PyModule::new(module_parent.py(), "many_bubbles_legacy")?;
    module_many_bubbles_legacy.add_class::<py_many_bubbles_legacy::py_bulk_flow::PyBulkFlow>()?;
    module_many_bubbles_legacy
        .add_class::<py_many_bubbles_legacy::py_bubble_formation::PyLattice>()?;
    module_many_bubbles_legacy
        .add_class::<py_many_bubbles_legacy::py_bubble_formation::PyPoissonNucleation>()?;
    module_many_bubbles_legacy
        .add_class::<py_many_bubbles_legacy::py_bubble_formation::PyManualNucleation>()?;
    module_many_bubbles_legacy
        .add_class::<py_many_bubbles_legacy::py_bubble_formation::PyBubbleFormationSimulator>()?;
    module_many_bubbles_legacy.add_function(wrap_pyfunction!(
        py_many_bubbles_legacy::py_bubble_formation::py_generate_bubbles_exterior,
        module_parent
    )?)?;

    let module_many_bubbles = PyModule::new(module_parent.py(), "many_bubbles")?;
    module_many_bubbles.add_class::<crate::py_many_bubbles::py_lattice::PyParallelepiped>()?;
    module_many_bubbles.add_class::<crate::py_many_bubbles::py_lattice::PyCartesian>()?;
    module_many_bubbles.add_class::<crate::py_many_bubbles::py_lattice::PySpherical>()?;
    module_many_bubbles.add_class::<crate::py_many_bubbles::py_lattice::PyEmpty>()?;
    module_many_bubbles.add_class::<py_many_bubbles::py_isometry::PyIsometry3>()?;
    module_many_bubbles
        .add_class::<py_many_bubbles::py_generalized_bulk_flow::PyGeneralizedBulkFlow>()?;
    module_many_bubbles.add_class::<py_many_bubbles::py_lattice_bubbles::PyLatticeBubbles>()?;
    module_many_bubbles
        .add_class::<py_many_bubbles::py_bubbles_nucleation::PyUniformAtFixedTime>()?;

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
