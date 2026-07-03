#pragma once
#include "slic3r/plugin/PythonPluginInterface.hpp"
#include "libslic3r/Print.hpp"      // SlicingPipelineStep
#include <pybind11/pybind11.h>
#include <string>

namespace Slic3r {

struct SlicingPipelineContext {
    std::string          orca_version;
    SlicingPipelineStep  step { SlicingPipelineStep::Slice };
    Print*               print  { nullptr };   // always present
    const PrintObject*   object { nullptr };   // null for print-wide steps
    // Read views (PrintView / PrintObjectView / …) are added in Task 8.
    bool cancelled() const;                      // -> print->canceled()
};

class SlicingPipelinePluginCapability : public PluginCapabilityInterface {
public:
    PluginCapabilityType get_type() const override { return PluginCapabilityType::SlicingPipeline; }
    virtual ExecutionResult execute(SlicingPipelineContext& ctx) = 0;
    static void RegisterBindings(pybind11::module_& module, pybind11::enum_<PluginCapabilityType>& pluginTypes);
};

} // namespace Slic3r
