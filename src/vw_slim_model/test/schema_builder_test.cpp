#include <gtest/gtest.h>

#include <vw_slim_model/schema_builder.hpp>
#include <vw_slim_model/vw_error.hpp>

namespace vw_slim_model_test {

TEST(VWModel, SchemaBuilder) {
    vw_slim_model::SchemaBuilder builder;
    auto result = builder.AddFloatFeature("MyInput1", 0, "MyNamespace");
    EXPECT_TRUE(result.has_value());
    // Using the same namespace is fine.
    result = builder.AddFloatFeature("MyInput2", "Hello", "MyNamespace");
    EXPECT_TRUE(result.has_value());
    // Using the same index is even fine.
    result = builder.AddFloatVectorFeature("MyInput3", 0, "MyNamespace");
    EXPECT_TRUE(result.has_value());
    // Using a previously used input name, however, is not fine.
    result = builder.AddFloatVectorFeature("MyInput2", 0, "AnotherNamespace");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(vw_slim_model::vw_errc::duplicate_input_name, result.error());
}
}  // namespace vw_slim_model_test
