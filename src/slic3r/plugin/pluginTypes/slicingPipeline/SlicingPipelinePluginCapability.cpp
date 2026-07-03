#include "SlicingPipelinePluginCapability.hpp"
#include "SlicingPipelinePluginCapabilityTrampoline.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;
namespace Slic3r {

bool SlicingPipelineContext::cancelled() const { return print && print->canceled(); }

void SlicingPipelinePluginCapability::RegisterBindings(py::module_& module, py::enum_<PluginCapabilityType>& pluginTypes) {
    (void) pluginTypes; // matches gcode/script/printerAgent; Step is a fresh enum below.
    auto slicing = module.def_submodule("slicing", "Slicing pipeline API (research/experimental).");

    py::enum_<SlicingPipelineStep>(slicing, "Step")
        .value("Slice", SlicingPipelineStep::Slice)
        .value("Perimeters", SlicingPipelineStep::Perimeters)
        .value("EstimateCurledExtrusions", SlicingPipelineStep::EstimateCurledExtrusions)
        .value("Infill", SlicingPipelineStep::Infill)          // fires after prepare+infill
        .value("Ironing", SlicingPipelineStep::Ironing)
        .value("Contouring", SlicingPipelineStep::Contouring)
        .value("SupportMaterial", SlicingPipelineStep::SupportMaterial)
        .value("DetectOverhangsForLift", SlicingPipelineStep::DetectOverhangsForLift)
        .value("SimplifyPath", SlicingPipelineStep::SimplifyPath) // covers all simplify sub-steps
        .value("WipeTower", SlicingPipelineStep::WipeTower)
        .value("SkirtBrim", SlicingPipelineStep::SkirtBrim)
        .export_values();

    py::class_<SlicingPipelineContext>(slicing, "SlicingPipelineContext")
        .def_readonly("orca_version", &SlicingPipelineContext::orca_version)
        .def_readonly("step", &SlicingPipelineContext::step)
        .def("cancelled", &SlicingPipelineContext::cancelled);
        // .object / read views added in Task 8

    py::class_<SlicingPipelinePluginCapability, PluginCapabilityInterface,
               PySlicingPipelinePluginCapabilityTrampoline,
               std::shared_ptr<SlicingPipelinePluginCapability>>(slicing, "SlicingPipelineCapabilityBase")
        .def(py::init<>());
}

} // namespace Slic3r
