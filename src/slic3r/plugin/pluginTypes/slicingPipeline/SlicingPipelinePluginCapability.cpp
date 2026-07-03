#include "SlicingPipelinePluginCapability.hpp"
#include "SlicingPipelinePluginCapabilityTrampoline.hpp"
#include "SlicingNumpy.hpp"          // make_readonly_rows
#include "libslic3r/libslic3r.h"    // unscale<>, live SCALING_FACTOR
#include <pybind11/stl.h>

namespace py = pybind11;
namespace Slic3r {

bool SlicingPipelineContext::cancelled() const { return print && print->canceled(); }

namespace {
// Zero-copy read-only int64 (N,2) view over a Polygon's points, pinned by `owner`.
// coord_t == int64; Point is asserted tightly packed in SlicingNumpy.hpp.
static py::array polygon_rows(const py::capsule& owner, const Polygon& poly)
{
    const Points& p = poly.points;
    return make_readonly_rows<coord_t, 2>(
        owner, p.empty() ? nullptr : p.front().data(), (py::ssize_t) p.size());
}
} // namespace

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

    // --- Read-graph geometry views (see header for the mandatory lifetime rule). ---
    // Every array/view below is valid ONLY during the execute(ctx) call that produced it.

    py::enum_<SurfaceType>(slicing, "SurfaceType")
        .value("stTop", stTop)
        .value("stBottom", stBottom)
        .value("stBottomBridge", stBottomBridge)
        .value("stInternalAfterExternalBridge", stInternalAfterExternalBridge)
        .value("stInternal", stInternal)
        .value("stInternalSolid", stInternalSolid)
        .value("stInternalBridge", stInternalBridge)
        .value("stSecondInternalBridge", stSecondInternalBridge)
        .value("stInternalVoid", stInternalVoid)
        .value("stPerimeter", stPerimeter)
        .value("stCount", stCount)
        .export_values();

    // Scaled integer coordinate -> millimeters. Reads the live SCALING_FACTOR at call
    // time (1e-6 normal, 1e-5 for beds > 2147mm), so it is never cached.
    slicing.def("unscale", [](coord_t v) { return unscale<double>(v); }, py::arg("coord"),
        "Convert a scaled integer coordinate to millimeters (reads the live SCALING_FACTOR).");

    py::class_<ExPolygonView>(slicing, "ExPolygonView")
        .def("contour", [](const ExPolygonView& v) { return polygon_rows(v.owner, v.ex->contour); },
            "Outer contour as a read-only int64 (N,2) numpy view in scaled coords. "
            "Valid only during the execute(ctx) call.")
        .def("holes", [](const ExPolygonView& v) {
            py::list out;
            for (const Polygon& h : v.ex->holes)
                out.append(polygon_rows(v.owner, h));
            return out;
        }, "List of hole contours (CW), each a read-only int64 (N,2) numpy view. "
           "Valid only during the execute(ctx) call.");

    py::class_<SurfaceView>(slicing, "SurfaceView")
        .def_property_readonly("surface_type",     [](const SurfaceView& v) { return v.s->surface_type; })
        .def_property_readonly("thickness",        [](const SurfaceView& v) { return v.s->thickness; })
        .def_property_readonly("bridge_angle",     [](const SurfaceView& v) { return v.s->bridge_angle; })
        .def_property_readonly("extra_perimeters", [](const SurfaceView& v) { return v.s->extra_perimeters; })
        .def_property_readonly("expolygon",        [](const SurfaceView& v) {
            return ExPolygonView{ &v.s->expolygon, v.owner };
        });

    py::class_<LayerRegionView>(slicing, "LayerRegionView")
        .def("slices", [](const LayerRegionView& v) {
            py::list out;
            for (const Surface& s : v.r->slices.surfaces)
                out.append(SurfaceView{ &s, v.owner });
            return out;
        }, "Sliced surfaces (typed top/bottom/internal) as [SurfaceView]. "
           "Valid only during the execute(ctx) call.")
        .def("fill_surfaces", [](const LayerRegionView& v) {
            py::list out;
            for (const Surface& s : v.r->fill_surfaces.surfaces)
                out.append(SurfaceView{ &s, v.owner });
            return out;
        }, "Surfaces prepared for infill as [SurfaceView]. "
           "Valid only during the execute(ctx) call.");

    py::class_<LayerView>(slicing, "LayerView")
        .def_property_readonly("slice_z", [](const LayerView& v) { return v.l->slice_z; })
        .def_property_readonly("print_z", [](const LayerView& v) { return v.l->print_z; })
        .def_property_readonly("height",  [](const LayerView& v) { return v.l->height; })
        .def("lslices", [](const LayerView& v) {
            py::list out;
            for (const ExPolygon& e : v.l->lslices)
                out.append(ExPolygonView{ &e, v.owner });
            return out;
        }, "Merged per-layer islands as [ExPolygonView]. "
           "Valid only during the execute(ctx) call.")
        .def("regions", [](const LayerView& v) {
            py::list out;
            for (const LayerRegion* r : v.l->regions())
                out.append(LayerRegionView{ r, v.owner });
            return out;
        }, "Per-region views as [LayerRegionView]. "
           "Valid only during the execute(ctx) call.");

    py::class_<PrintObjectView>(slicing, "PrintObjectView")
        .def("layers", [](const PrintObjectView& v) {
            py::list out;
            for (const Layer* l : v.o->layers())
                out.append(LayerView{ l, v.owner });
            return out;
        }, "Object layers as [LayerView]. Valid only during the execute(ctx) call.");

    py::class_<SlicingPipelineContext>(slicing, "SlicingPipelineContext")
        .def_readonly("orca_version", &SlicingPipelineContext::orca_version)
        .def_readonly("step", &SlicingPipelineContext::step)
        .def_property_readonly("object", [](const SlicingPipelineContext& ctx) -> py::object {
            if (ctx.object == nullptr)
                return py::none();
            return py::cast(PrintObjectView{ ctx.object, ctx.owner });
        }, "PrintObjectView for object-scoped steps, or None for print-wide steps. "
           "Valid only during the execute(ctx) call.")
        .def("cancelled", &SlicingPipelineContext::cancelled);

    py::class_<SlicingPipelinePluginCapability, PluginCapabilityInterface,
               PySlicingPipelinePluginCapabilityTrampoline,
               std::shared_ptr<SlicingPipelinePluginCapability>>(slicing, "SlicingPipelineCapabilityBase")
        .def(py::init<>())
        .def("get_type", &SlicingPipelinePluginCapability::get_type)
        .def("execute", &SlicingPipelinePluginCapability::execute);
}

} // namespace Slic3r
