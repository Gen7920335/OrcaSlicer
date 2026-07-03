#include <catch2/catch_test_macros.hpp>
#include "libslic3r/PrintConfig.hpp"
using namespace Slic3r;

TEST_CASE("slicing_pipeline_plugin option exists and defaults empty", "[slicing_pipeline]") {
    DynamicPrintConfig cfg = DynamicPrintConfig::full_print_config();
    const ConfigOptionStrings* opt = cfg.option<ConfigOptionStrings>("slicing_pipeline_plugin");
    REQUIRE(opt != nullptr);
    CHECK(opt->values.empty());
    const ConfigOptionDef* def = cfg.def()->get("slicing_pipeline_plugin");
    REQUIRE(def != nullptr);
    CHECK(def->support_plugin == true);
    CHECK(def->gui_type == ConfigOptionDef::GUIType::plugin_picker);
}

#include "libslic3r/Print.hpp"

TEST_CASE("slicing pipeline hook setter is a no-op-safe injection", "[slicing_pipeline]") {
    int calls = 0;
    Slic3r::Print::set_slicing_pipeline_hook_fn(
        [&](Slic3r::Print&, const Slic3r::PrintObject*, Slic3r::SlicingPipelineStep){ ++calls; });
    Slic3r::Print::set_slicing_pipeline_hook_fn(nullptr); // reset — must be legal
    CHECK(calls == 0);
}
