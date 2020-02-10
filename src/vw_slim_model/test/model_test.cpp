#include <gtest/gtest.h>

#include <vw_slim_model/model.hpp>
#include <vw_slim_model/schema_builder.hpp>
#include <vw_slim_model/vw_error.hpp>

#include "env.hpp"

namespace vw_slim_model_test {
TEST(VWModel, Model) {
    auto model_path = test_dir_path + "slimdata/regression_data_3.model";
    auto model_data = all_bytes(model_path);

    vw_slim_model::SchemaBuilder sb;
    auto result = sb.AddFloatFeature("InputA", 0, "a");
    ASSERT_TRUE(result.has_value());
    result = sb.AddFloatFeature("InputB", 2, "b");
    ASSERT_TRUE(result.has_value());

    auto model = vw_slim_model::Model::Load(sb, model_data).value_or(nullptr);
    ASSERT_NE(nullptr, model);
}
}  // namespace vw_slim_model_test