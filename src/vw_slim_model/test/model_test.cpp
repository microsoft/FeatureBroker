// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <inference/feature_broker.hpp>
#include <vw_common/error.hpp>
#include <vw_common/schema_builder.hpp>
#include <vw_slim_model/model.hpp>

#include "data.h"
#include "env.hpp"

namespace resonance_vw_test {

TEST(VWModel, Model) {
    auto model_path = test_dir_path + "slimdata/regression_data_3.model";
    auto model_data = all_bytes(model_path);

    resonance_vw::SchemaBuilder sb;
    auto result = sb.AddFloatFeature("InputA", 0, "a");
    ASSERT_TRUE(result.has_value());
    result = sb.AddFloatFeature("InputB", 2, "b");
    ASSERT_TRUE(result.has_value());

    auto task = resonance_vw::OutputTask::MakeRegression("Output");
    auto model = resonance_vw::Model::Load(sb, task, model_data).value_or(nullptr);
    ASSERT_NE(nullptr, model);

    inference::TypeDescriptor floatType = inference::TypeDescriptor::Create<float>();
    ASSERT_EQ(2, model->Inputs().size());
    auto foundType = model->Inputs().find("InputA");
}

TEST(VWModel, ModelUse) {
    auto model_path = test_dir_path + "slimdata/regression_data_3.model";
    auto model_data = all_bytes(model_path);

    resonance_vw::SchemaBuilder sb;
    auto result = sb.AddFloatFeature("InputA", 0, "a");
    ASSERT_TRUE(result.has_value());
    result = sb.AddFloatFeature("InputB", 2, "b");
    ASSERT_TRUE(result.has_value());

    auto task = resonance_vw::OutputTask::MakeRegression("Output");
    auto model = resonance_vw::Model::Load(sb, task, model_data).value_or(nullptr);
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
    ASSERT_TRUE(updateExpected.has_value());
    ASSERT_TRUE(updateExpected.value());
    // Should be about 0.804214, going by regression_data_3.pred. Give it wiggle room on the last place.
    ASSERT_NEAR(0.804214, score, 1e-6);

    inputB->Feed(4.f);
    updateExpected = output->UpdateIfChanged(score);
    ASSERT_TRUE(updateExpected.has_value());
    ASSERT_TRUE(updateExpected.value());
    ASSERT_NEAR(0.119599, score, 1e-6);
}
}  // namespace resonance_vw_test
