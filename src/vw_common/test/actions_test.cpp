
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <vw_common/actions.hpp>
#include <vw_common/error.hpp>

using namespace resonance_vw;

namespace resonance_vw_test {
TEST(vw_common, Actions) {
    auto result = Actions::Create(std::vector<int>{1, 2, 3, 4});
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value()->Type(), ActionType::Int);
    auto resultIntVec = result.value()->GetIntActions();
    EXPECT_TRUE(resultIntVec.has_value());
    EXPECT_EQ(resultIntVec.value().size(), 4);

    result = Actions::Create(std::vector<float>{1.1f, 2.3f, 4.5f});
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value()->Type(), ActionType::Float);
    auto resultFloatVec = result.value()->GetFloatActions();
    EXPECT_TRUE(resultFloatVec.has_value());
    EXPECT_EQ(resultFloatVec.value().size(), 3);

    result = Actions::Create(std::vector<std::string>{"1", "2", "3", "4"});
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value()->Type(), ActionType::String);
    auto resultStringVec = result.value()->GetStringActions();
    EXPECT_TRUE(resultStringVec.has_value());
    EXPECT_EQ(resultStringVec.value().size(), 4);
}

TEST(vw_common, ActionsBadCase) {
    auto result = Actions::Create(std::vector<int>{});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(vw_errc::invalid_actions, result.error());

    result = Actions::Create(std::vector<float>{});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(vw_errc::invalid_actions, result.error());

    result = Actions::Create(std::vector<std::string>{});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(vw_errc::invalid_actions, result.error());

    // Invalid type
    result = Actions::Create(std::vector<double>{});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(vw_errc::invalid_actions, result.error());
}

}  // namespace resonance_vw_test
