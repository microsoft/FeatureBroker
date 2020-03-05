// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <inference/feature_broker.hpp>
#include <vw_common/actions.hpp>
#include <vw_common/error.hpp>
#include <vw_common/schema_builder.hpp>
#include <vw_remote_model/cpprest_recommender_client.hpp>
#include <vw_remote_model/remote_model.hpp>

#include "env.hpp"

static std::string apsBaseUri = "ENTER_URI";
static std::string apsSubscriptionKey = "ENTER_KEY";

using namespace resonance_vw;

namespace resonance_vw_test {
TEST(VWModel, Model) {
    resonance_vw::SchemaBuilder sb;
    auto result = sb.AddFloatFeature("InputA", 0, "a");
    ASSERT_TRUE(result.has_value());
    result = sb.AddFloatFeature("InputB", 2, "b");
    ASSERT_TRUE(result.has_value());

    auto actions = std::vector<std::string>{"1", "2", "3", "4"};
    auto actionsResult = resonance_vw::Actions::Create(actions);
    ASSERT_TRUE(actionsResult.has_value());
    auto client = CppRestRecommenderClient::Create(apsBaseUri, apsSubscriptionKey).value_or(nullptr);
    ASSERT_NE(nullptr, client);
    auto model = RemoteModel::Load(sb, actionsResult.value(), client).value_or(nullptr);
    ASSERT_NE(nullptr, model);
}

TEST(VWModel, ModelUse) {
    resonance_vw::SchemaBuilder sb;
    auto result = sb.AddFloatFeature("InputA", "InputA", "a");
    ASSERT_TRUE(result.has_value());
    result = sb.AddFloatFeature("InputB", "InputB", "a");
    ASSERT_TRUE(result.has_value());

    auto actions = std::vector<float>{1.f, 2.f, 3.f, 4.f};
    auto actionsResult = resonance_vw::Actions::Create(actions);
    ASSERT_TRUE(actionsResult.has_value());

    // Create the recommendation client
    auto client = CppRestRecommenderClient::Create(apsBaseUri, apsSubscriptionKey).value_or(nullptr);
    ASSERT_NE(nullptr, client);
    auto model = RemoteModel::Load(sb, actionsResult.value(), client).value_or(nullptr);
    ASSERT_NE(nullptr, model);

    inference::FeatureBroker fb(model);
    auto inputA = fb.BindInput<float>("InputA").value_or(nullptr);
    ASSERT_NE(nullptr, inputA);
    auto inputB = fb.BindInput<float>("InputB").value_or(nullptr);
    ASSERT_NE(nullptr, inputB);

    auto output = fb.BindOutput<float>("Output").value_or(nullptr);
    ASSERT_NE(nullptr, output);

    inputA->Feed(1.f);
    inputB->Feed(2.f);
    float score = 0;
    auto updateExpected = output->UpdateIfChanged(score);
    ASSERT_EQ(3.f, score);
}
}  // namespace resonance_vw_test