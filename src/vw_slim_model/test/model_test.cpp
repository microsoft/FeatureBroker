// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <inference/feature_broker.hpp>
#include <vw_common/error.hpp>
#include <vw_common/schema_builder.hpp>
#include <vw_slim_model/model.hpp>

#include "env.hpp"

namespace {
#include "data.h"
}

namespace resonance_vw_test {

TEST(VWModel, Model) {
    auto model_path = test_dir_path + "slimdata/regression_data_3.model";
    auto model_data = all_bytes(model_path);

    resonance_vw::SchemaBuilder sb;
    auto result = sb.AddFloatFeature("InputA", 0, "a");
    ASSERT_TRUE(result.has_value());
    result = sb.AddFloatFeature("InputB", 2, "b");
    ASSERT_TRUE(result.has_value());

    auto task = resonance_vw::OutputTask::MakeRegression();
    auto model = resonance_vw::Model::Load(sb, task, model_data.data(), model_data.size()).value_or(nullptr);
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

    auto task = resonance_vw::OutputTask::MakeRegression();
    auto model = resonance_vw::Model::Load(sb, task, model_data.data(), model_data.size()).value_or(nullptr);
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

TEST(VWModel, RecommenderModelUse) {
    resonance_vw::SchemaBuilder sb;
    auto result = sb.AddIntFeature("CallType", 0, "64");
    ASSERT_TRUE(result.has_value());
    result = sb.AddIntFeature("Modality", 0, "16");
    ASSERT_TRUE(result.has_value());
    result = sb.AddIntFeature("NetworkType", 0, "32");
    ASSERT_TRUE(result.has_value());
    result = sb.AddIntFeature("Platform", 0, "48");
    ASSERT_TRUE(result.has_value());

    auto action = resonance_vw::Actions::Create<std::string>({"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"});
    ASSERT_TRUE(action);
    auto task = resonance_vw::OutputTask::MakeRecommendation(action.value(), "eid", "80");
    ASSERT_TRUE(task);
    auto model = resonance_vw::Model::Load(sb, task.value(), ::cb_data_epsilon_0_skype_jb_model,
                                           sizeof(::cb_data_epsilon_0_skype_jb_model))
                     .value_or(nullptr);
    ASSERT_NE(nullptr, model);

    inference::FeatureBroker fb(model);
    auto inputA = fb.BindInput<int>("CallType").value_or(nullptr);
    ASSERT_NE(nullptr, inputA);
    auto inputB = fb.BindInput<int>("Modality").value_or(nullptr);
    ASSERT_NE(nullptr, inputB);
    auto inputC = fb.BindInput<int>("NetworkType").value_or(nullptr);
    ASSERT_NE(nullptr, inputC);
    auto inputD = fb.BindInput<int>("Platform").value_or(nullptr);
    ASSERT_NE(nullptr, inputD);

    // In principle this should be a multi-output single bind, but this helps us exercise some of the other aspects of
    // the plumbing.
    auto outputA = fb.BindOutput<inference::Tensor<std::string>>("Actions").value_or(nullptr);
    ASSERT_NE(nullptr, outputA);
    auto outputI = fb.BindOutput<inference::Tensor<int>>("Indices").value_or(nullptr);
    ASSERT_NE(nullptr, outputI);
    auto outputP = fb.BindOutput<inference::Tensor<float>>("Probabilities").value_or(nullptr);
    ASSERT_NE(nullptr, outputP);

    inputA->Feed(0);
    inputB->Feed(1);
    inputC->Feed(2);
    inputD->Feed(3);

    ASSERT_TRUE(outputA->Changed());

    inference::Tensor<std::string> actions;
    inference::Tensor<int> indices;
    inference::Tensor<float> pdf;

    auto updateExpected = outputA->UpdateIfChanged(actions);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    updateExpected = outputI->UpdateIfChanged(indices);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    updateExpected = outputP->UpdateIfChanged(pdf);
    ASSERT_TRUE(updateExpected && updateExpected.value());

    std::vector<size_t> dims({10});
    ASSERT_EQ(dims, actions.Dimensions());
    ASSERT_EQ(dims, indices.Dimensions());
    ASSERT_EQ(dims, pdf.Dimensions());
}
}  // namespace resonance_vw_test
