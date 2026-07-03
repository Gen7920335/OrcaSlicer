#include <catch2/catch_test_macros.hpp>
#include "slic3r/plugin/PythonPluginInterface.hpp"
using namespace Slic3r;

TEST_CASE("SlicingPipeline capability-type string maps round-trip", "[slicing_pipeline]") {
    CHECK(plugin_capability_type_to_string(PluginCapabilityType::SlicingPipeline) == "slicing-pipeline");
    CHECK(plugin_capability_type_display_name(PluginCapabilityType::SlicingPipeline) == "Slicing Pipeline");
    CHECK(plugin_capability_type_from_string("slicing-pipeline") == PluginCapabilityType::SlicingPipeline);
    CHECK(plugin_capability_type_from_string("SLICING-PIPELINE") == PluginCapabilityType::SlicingPipeline);
    CHECK(plugin_capability_type_from_string("nope") == PluginCapabilityType::Unknown);
}

#include "python_test_support.hpp"
#include "slic3r/plugin/pluginTypes/slicingPipeline/SlicingNumpy.hpp"
#include "libslic3r/Point.hpp"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

TEST_CASE("make_readonly_rows builds a read-only (N,2) int64 view", "[slicing_pipeline]") {
    ensure_python_initialized(); // helper already used by test_plugin_host_api.cpp
    py::gil_scoped_acquire gil;

    // make_readonly_rows() constructs a py::array_t, which requires numpy to be
    // importable in the embedded interpreter. The unit-test interpreter ships no
    // site-packages (same condition test_plugin_host_api.cpp's TriangleMesh numpy
    // test guards against), so skip the array-backed assertions when numpy is
    // unavailable there rather than fail on an environment quirk.
    bool have_numpy = false;
    try {
        py::module_::import("numpy");
        have_numpy = true;
    } catch (const py::error_already_set&) {
        have_numpy = false;
    }
    if (!have_numpy) {
        SKIP("numpy unavailable in unit-test interpreter");
    }

    static Slic3r::Points pts = { Slic3r::Point(10, 20), Slic3r::Point(30, 40) };
    py::capsule keepalive(&pts, [](void*){});
    py::array a = Slic3r::make_readonly_rows<coord_t, 2>(keepalive, pts.front().data(), (py::ssize_t)pts.size());
    CHECK(a.dtype().kind() == 'i');
    CHECK(a.itemsize() == 8);           // int64
    CHECK(a.shape(0) == 2);
    CHECK(a.shape(1) == 2);
    CHECK_FALSE(a.writeable());
    auto r = a.unchecked<coord_t, 2>();
    CHECK(r(0,0) == 10); CHECK(r(1,1) == 40);
}
