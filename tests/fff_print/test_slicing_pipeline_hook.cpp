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
