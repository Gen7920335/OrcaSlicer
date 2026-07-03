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
#include "slic3r/plugin/pluginTypes/slicingPipeline/SlicingPipelinePluginCapability.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Surface.hpp"
#include "libslic3r/Layer.hpp"
#include "libslic3r/ExtrusionEntity.hpp"
#include "libslic3r/ExtrusionEntityCollection.hpp"
#include <catch2/matchers/catch_matchers_floating_point.hpp>
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

TEST_CASE("orca.slicing module: Step enum, context, and a Python capability can execute", "[slicing_pipeline]") {
    ensure_python_initialized();
    import_orca_module(); // forces PythonPluginBridge::instance() (see test_plugin_host_api.cpp:32-40)
    py::gil_scoped_acquire gil;
    py::module_ orca = py::module_::import("orca");
    REQUIRE(py::hasattr(orca, "slicing"));
    py::object slicing = orca.attr("slicing");
    CHECK(py::hasattr(slicing, "Step"));
    CHECK(py::hasattr(slicing.attr("Step"), "Slice"));
    CHECK(py::hasattr(slicing, "SlicingPipelineContext"));
    CHECK(py::hasattr(slicing, "SlicingPipelineCapabilityBase"));

    // A trivial Python subclass whose execute() reports success, invoked via the C++ trampoline.
    py::exec(R"(
import orca
class Probe(orca.slicing.SlicingPipelineCapabilityBase):
    def get_name(self): return "probe"
    def execute(self, ctx): return orca.ExecutionResult.success("ok")
_probe = Probe()
    )");
    // (Full C++ trampoline invocation with a real context is exercised in Task 8's tests.)
}

// Numpy-free half of Task 8: type registration, the SurfaceType enum, the module-level
// unscale() helper, and every non-array read accessor (surface_type / thickness /
// bridge_angle / extra_perimeters / expolygon / empty holes()). None of these
// materialize a py::array, so they run unconditionally (no numpy guard needed).
TEST_CASE("orca.slicing geometry views: types, SurfaceType, unscale, non-array accessors", "[slicing_pipeline]") {
    using Catch::Matchers::WithinRel;
    using Catch::Matchers::WithinAbs;
    ensure_python_initialized();
    import_orca_module();
    py::gil_scoped_acquire gil;
    py::object slicing = py::module_::import("orca").attr("slicing");

    // All view types are registered in the submodule.
    for (const char* name : { "ExPolygonView", "SurfaceView", "LayerRegionView",
                              "LayerView", "PrintObjectView", "SurfaceType" })
        CHECK(py::hasattr(slicing, name));

    // Read-graph traversal methods exist on the class objects (verified without a
    // full Print, which slic3rutils cannot build).
    CHECK(py::hasattr(slicing.attr("ExPolygonView"), "contour"));
    CHECK(py::hasattr(slicing.attr("ExPolygonView"), "holes"));
    CHECK(py::hasattr(slicing.attr("LayerRegionView"), "slices"));
    CHECK(py::hasattr(slicing.attr("LayerRegionView"), "fill_surfaces"));
    CHECK(py::hasattr(slicing.attr("LayerView"), "regions"));
    CHECK(py::hasattr(slicing.attr("LayerView"), "lslices"));
    CHECK(py::hasattr(slicing.attr("PrintObjectView"), "layers"));
    CHECK(py::hasattr(slicing.attr("SlicingPipelineContext"), "object"));

    // SurfaceType enum values round-trip to the C++ enumerators.
    py::object ST = slicing.attr("SurfaceType");
    CHECK(ST.attr("stTop").cast<Slic3r::SurfaceType>()           == Slic3r::stTop);
    CHECK(ST.attr("stInternalSolid").cast<Slic3r::SurfaceType>() == Slic3r::stInternalSolid);
    CHECK(ST.attr("stPerimeter").cast<Slic3r::SurfaceType>()     == Slic3r::stPerimeter);
    CHECK(ST.attr("stCount").cast<Slic3r::SurfaceType>()         == Slic3r::stCount);

    // unscale() reads the live SCALING_FACTOR both when scaling and unscaling.
    const coord_t scaled10 = (coord_t) scale_(10.0);
    double mm = slicing.attr("unscale")(scaled10).cast<double>();
    CHECK_THAT(mm, WithinRel(10.0, 1e-9));

    // SurfaceView non-array accessors against a hand-built Surface.
    Slic3r::Surface surf(Slic3r::stInternalSolid);
    surf.thickness        = 0.4;
    surf.bridge_angle     = -1.0;
    surf.extra_perimeters = 2;
    py::capsule owner(&surf, [](void*){});   // no-op owner (data outlives the view here)
    py::object sv = py::cast(Slic3r::SurfaceView{ &surf, owner });
    CHECK(sv.attr("surface_type").cast<Slic3r::SurfaceType>() == Slic3r::stInternalSolid);
    CHECK_THAT(sv.attr("thickness").cast<double>(), WithinRel(0.4, 1e-9));
    CHECK_THAT(sv.attr("bridge_angle").cast<double>(), WithinAbs(-1.0, 1e-12));
    CHECK(sv.attr("extra_perimeters").cast<int>() == 2);

    // expolygon accessor yields an ExPolygonView; holes() on an empty ExPolygon is an
    // empty list and materializes no array (so it stays outside the numpy guard).
    py::object exv = sv.attr("expolygon");
    CHECK(py::hasattr(exv, "contour"));
    CHECK(exv.attr("holes")().cast<py::list>().size() == 0);
}

TEST_CASE("ExPolygonView.contour()/holes() are read-only int64 (N,2) views in scaled coords", "[slicing_pipeline]") {
    ensure_python_initialized();
    import_orca_module();
    py::gil_scoped_acquire gil;

    // make_readonly_rows() constructs a py::array, which needs numpy at runtime; the
    // unit-test interpreter ships none. Skip the array-backed assertions when numpy is
    // unavailable (same convention as the make_readonly_rows test above).
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

    const coord_t s = (coord_t) scale_(10.0);
    Slic3r::ExPolygon ex;
    ex.contour.points = { Slic3r::Point(0, 0), Slic3r::Point(s, 0),
                          Slic3r::Point(s, s), Slic3r::Point(0, s) };
    Slic3r::Polygon hole;
    hole.points = { Slic3r::Point(1, 1), Slic3r::Point(2, 1), Slic3r::Point(2, 2) };
    ex.holes = { hole };

    py::capsule owner(&ex, [](void*){});
    py::object view = py::cast(Slic3r::ExPolygonView{ &ex, owner });

    py::array c = view.attr("contour")().cast<py::array>();
    CHECK(c.dtype().kind() == 'i');
    CHECK(c.itemsize() == 8);           // int64
    CHECK(c.shape(0) == 4);
    CHECK(c.shape(1) == 2);
    CHECK_FALSE(c.writeable());
    auto rc = c.cast<py::array_t<coord_t>>().unchecked<2>();
    CHECK(rc(0, 0) == 0);
    CHECK(rc(1, 0) == s);
    CHECK(rc(2, 1) == s);

    // holes() -> list of read-only (N,2) int64 views.
    py::list holes = view.attr("holes")().cast<py::list>();
    CHECK(holes.size() == 1);
    py::array h0 = holes[0].cast<py::array>();
    CHECK(h0.shape(0) == 3);
    CHECK(h0.shape(1) == 2);
    CHECK_FALSE(h0.writeable());
}

// ---------------------------------------------------------------------------
// Task 9: toolpaths (PathData over perimeters/fills).
//
// LayerRegion's ctor is protected (constructed only by Layer/PrintObject). A
// trivial derived struct lets a unit test build one with null layer/region
// pointers — perimeters()/fills() only read the public `perimeters`/`fills`
// collections, never the layer/region back-pointers.
// ---------------------------------------------------------------------------
namespace {
struct TestLayerRegion : Slic3r::LayerRegion {
    TestLayerRegion() : Slic3r::LayerRegion(nullptr, nullptr) {}
};

// Build a realistic nested perimeters collection into `region.perimeters`:
//   perimeters (outer) -> inner collection -> [ ExtrusionLoop(pathA), ExtrusionPath(pathB) ]
// This exercises both the recursive descent through nested collections and the
// decomposition of an ExtrusionLoop into its contained ExtrusionPath (flatten()
// does NOT decompose loops, hence the hand-rolled recursive walk).
static void build_nested_perimeters(TestLayerRegion& region) {
    using namespace Slic3r;
    ExtrusionPath pathA(erExternalPerimeter);        // -> "Outer wall"
    pathA.mm3_per_mm = 0.05; pathA.width = 0.45f; pathA.height = 0.20f;
    pathA.polyline.points = { Point3(0, 0, 0), Point3(10, 0, 0), Point3(10, 10, 0) };

    ExtrusionPath pathB(erInternalInfill);           // -> "Sparse infill"
    pathB.mm3_per_mm = 0.03; pathB.width = 0.40f; pathB.height = 0.20f;
    pathB.polyline.points = { Point3(1, 1, 0), Point3(2, 1, 0), Point3(2, 2, 0) };

    ExtrusionEntityCollection inner;
    inner.append(ExtrusionLoop(pathA));              // clone_move
    inner.append(pathB);                             // clone
    region.perimeters.append(inner);                 // nested (deep clone)
}
} // namespace

// Numpy-free half: perimeters() flattens the nested graph (descending through
// collections and decomposing loops) into a [PathData] list; role/width/height/
// mm3_per_mm are plain scalars, so these assertions run unconditionally.
TEST_CASE("orca.slicing LayerRegionView.perimeters()/fills(): PathData scalars over a nested graph", "[slicing_pipeline]") {
    using Catch::Matchers::WithinRel;
    ensure_python_initialized();
    import_orca_module();
    py::gil_scoped_acquire gil;
    py::object slicing = py::module_::import("orca").attr("slicing");

    CHECK(py::hasattr(slicing, "PathData"));
    CHECK(py::hasattr(slicing.attr("LayerRegionView"), "perimeters"));
    CHECK(py::hasattr(slicing.attr("LayerRegionView"), "fills"));

    TestLayerRegion region;
    build_nested_perimeters(region);
    py::capsule owner(&region, [](void*){});         // no-op: region outlives the view
    py::object lrv = py::cast(Slic3r::LayerRegionView{ &region, owner });

    py::list ps = lrv.attr("perimeters")().cast<py::list>();
    REQUIRE(ps.size() == 2);                          // loop's path + bare path

    py::object pd0 = ps[0];                           // pathA, from the loop
    CHECK(pd0.attr("role").cast<std::string>() == "Outer wall");
    CHECK_THAT(pd0.attr("width").cast<double>(),      WithinRel(0.45, 1e-6));
    CHECK_THAT(pd0.attr("height").cast<double>(),     WithinRel(0.20, 1e-6));
    CHECK_THAT(pd0.attr("mm3_per_mm").cast<double>(), WithinRel(0.05, 1e-9));

    py::object pd1 = ps[1];                           // pathB, bare
    CHECK(pd1.attr("role").cast<std::string>() == "Sparse infill");
    CHECK_THAT(pd1.attr("width").cast<double>(),      WithinRel(0.40, 1e-6));

    // fills is empty on this hand-built region.
    CHECK(lrv.attr("fills")().cast<py::list>().size() == 0);
}

// Numpy-backed half: PathData.points() materializes a read-only (N,3) int64 view.
TEST_CASE("orca.slicing PathData.points() is a read-only (N,3) int64 view", "[slicing_pipeline]") {
    ensure_python_initialized();
    import_orca_module();
    py::gil_scoped_acquire gil;

    // make_readonly_rows() needs numpy at runtime; the unit-test interpreter ships
    // none. Skip the array-backed assertions when numpy is unavailable (same
    // convention as the make_readonly_rows / ExPolygonView tests above).
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

    TestLayerRegion region;
    build_nested_perimeters(region);
    py::capsule owner(&region, [](void*){});
    py::object lrv = py::cast(Slic3r::LayerRegionView{ &region, owner });

    py::list ps = lrv.attr("perimeters")().cast<py::list>();
    REQUIRE(ps.size() == 2);

    // pathB has 3 points: (1,1,0), (2,1,0), (2,2,0).
    py::array pts = ps[1].attr("points")().cast<py::array>();
    CHECK(pts.dtype().kind() == 'i');
    CHECK(pts.itemsize() == 8);                       // int64
    CHECK(pts.shape(0) == 3);
    CHECK(pts.shape(1) == 3);
    CHECK_FALSE(pts.writeable());
    auto r = pts.cast<py::array_t<coord_t>>().unchecked<2>();
    CHECK(r(0, 0) == 1); CHECK(r(0, 1) == 1); CHECK(r(0, 2) == 0);
    CHECK(r(1, 0) == 2);
    CHECK(r(2, 1) == 2);
}
