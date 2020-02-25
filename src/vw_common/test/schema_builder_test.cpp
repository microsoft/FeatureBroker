// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <vw_common/error.hpp>
#include <vw_common/schema_builder.hpp>

using namespace resonance_vw;

namespace resonance_vw_test {

TEST(vw_common, SchemaBuilder) {
    SchemaBuilder builder;
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
    EXPECT_EQ(vw_errc::duplicate_input_name, result.error());
}
}  // namespace resonance_vw_test
