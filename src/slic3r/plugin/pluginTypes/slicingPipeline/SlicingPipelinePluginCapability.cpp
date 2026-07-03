#include "SlicingPipelinePluginCapability.hpp"
#include "SlicingPipelinePluginCapabilityTrampoline.hpp"
#include "SlicingNumpy.hpp"          // make_readonly_rows
#include "libslic3r/libslic3r.h"    // unscale<>, live SCALING_FACTOR
#include "libslic3r/ExtrusionEntity.hpp"            // ExtrusionPath/Loop/MultiPath, role_to_string
#include "libslic3r/ExtrusionEntityCollection.hpp"  // ExtrusionEntityCollection
#include <pybind11/stl.h>
#include <vector>

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

// Flatten an extrusion graph into a list of leaf ExtrusionPath* while walking the
// ORIGINAL Print-owned tree (never a temporary copy): the returned pointers stay
// valid for the execute(ctx) lifetime pinned by `owner`, so points() can hand out
// zero-copy views into path->polyline.points.
//
// This is deliberately NOT ExtrusionEntityCollection::flatten(): flatten() only
// unwraps nested collections (is_collection() is true solely for collections) and
// returns them by value, so it would (a) dangle if we viewed into the copy and
// (b) leave ExtrusionLoop/ExtrusionMultiPath intact — dropping every perimeter
// loop, since dynamic_cast<ExtrusionPath*> fails on a loop. We descend into
// loops/multipaths here to reach their contained paths.
static void collect_extrusion_paths(const ExtrusionEntity* ee, std::vector<const ExtrusionPath*>& out)
{
    if (ee == nullptr)
        return;
    if (const auto* coll = dynamic_cast<const ExtrusionEntityCollection*>(ee)) {
        for (const ExtrusionEntity* child : coll->entities)
            collect_extrusion_paths(child, out);
    } else if (const auto* loop = dynamic_cast<const ExtrusionLoop*>(ee)) {
        for (const ExtrusionPath& p : loop->paths)
            out.push_back(&p);
    } else if (const auto* mp = dynamic_cast<const ExtrusionMultiPath*>(ee)) {
        for (const ExtrusionPath& p : mp->paths)
            out.push_back(&p);
    } else if (const auto* path = dynamic_cast<const ExtrusionPath*>(ee)) {
        // Catches ExtrusionPath and its subclasses (Sloped/Contoured/Oriented) last,
        // after the composite types above have been ruled out.
        out.push_back(path);
    }
}

// Build a Python list of PathData over an extrusion collection, each entry pinned by `owner`.
static py::list path_data_list(const py::capsule& owner, const ExtrusionEntityCollection& coll)
{
    std::vector<const ExtrusionPath*> paths;
    collect_extrusion_paths(&coll, paths);
    py::list out;
    for (const ExtrusionPath* p : paths)
        out.append(PathData{ p, owner });
    return out;
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

    // A flattened toolpath. Read-only in v1 (mutation is a later phase). role/width/
    // height/mm3_per_mm are plain scalars; points() materializes a zero-copy array.
    py::class_<PathData>(slicing, "PathData")
        .def("points", [](const PathData& p) {
            const Points3& pts = p.path->polyline.points;
            return make_readonly_rows<coord_t, 3>(
                p.owner, pts.empty() ? nullptr : pts.front().data(), (py::ssize_t) pts.size());
        }, "Path vertices as a read-only int64 (N,3) numpy view in scaled coords "
           "(the polyline is natively 3D on this branch). Valid only during the execute(ctx) call.")
        .def_property_readonly("role", [](const PathData& p) {
            return ExtrusionEntity::role_to_string(p.path->role());
        }, "Extrusion role as a human-readable string (e.g. \"Outer wall\", \"Sparse infill\").")
        .def_property_readonly("width",      [](const PathData& p) { return p.path->width; })
        .def_property_readonly("height",     [](const PathData& p) { return p.path->height; })
        .def_property_readonly("mm3_per_mm", [](const PathData& p) { return p.path->mm3_per_mm; });

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
           "Valid only during the execute(ctx) call.")
        .def("perimeters", [](const LayerRegionView& v) {
            return path_data_list(v.owner, v.r->perimeters);
        }, "Perimeter toolpaths flattened to [PathData] (nested collections and "
           "loops decomposed into their paths). Valid only during the execute(ctx) call.")
        .def("fills", [](const LayerRegionView& v) {
            return path_data_list(v.owner, v.r->fills);
        }, "Infill toolpaths flattened to [PathData] (nested collections and loops "
           "decomposed into their paths). Valid only during the execute(ctx) call.");

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
