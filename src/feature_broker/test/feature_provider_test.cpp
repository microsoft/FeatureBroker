// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <inference/feature_broker.hpp>
#include <inference/synchronous_feature_broker.hpp>
#include <memory>
#include <string>
#include <tuple>

#include "add_five_model.hpp"
#include "gtest/gtest.h"
#include "three_output_model.hpp"
#include "tuple_feature_providers.hpp"

using namespace ::inference;

TEST(InferenceTestSuite, ProviderInputAndOutput) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto fp = inference_test::TupleProviderFactory::Create<float>("A");
    auto bindExpected = fb->BindInputs(fp);
    ASSERT_TRUE((bool)bindExpected);

    float value = 0;
    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_FALSE(output->Changed());
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    fp->Set<0>(2.0f);
    ASSERT_TRUE(output->Changed());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);
    ASSERT_FALSE(output->Changed());
}

TEST(InferenceTestSuite, ProviderSyncInputAndOutput) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<SynchronousFeatureBroker>(model);

    auto fp = inference_test::TupleProviderFactory::Create<float>("A");
    auto bindExpected = fb->BindInputs(fp);
    ASSERT_TRUE((bool)bindExpected);

    float value;
    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    fp->Set<0>(2.0f);
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);
}

TEST(InferenceTestSuite, ProviderMultiInputAndOutput) {
    std::shared_ptr<Model> model(new inference_test::ThreeOutputModel());
    auto fb = std::make_shared<FeatureBroker>(model);

    auto fp = inference_test::TupleProviderFactory::Create<int, float>("A", "B");
    auto bindExpected = fb->BindInputs(fp);
    ASSERT_TRUE((bool)bindExpected);

    fp->Set<0>(4);
    fp->Set<1>(3.0f);

    auto outputExp = fb->BindOutputs<int, std::string>({"X", "Z"});
    ASSERT_TRUE(outputExp && outputExp.value());
    auto output = outputExp.value();

    std::tuple<int, std::string> values;
    auto updateExpected = output->UpdateIfChanged(values);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(9, std::get<0>(values));
}

TEST(InferenceTestSuite, ProviderAndPipe) {
    std::shared_ptr<Model> model(new inference_test::ThreeOutputModel());
    auto fb = std::make_shared<FeatureBroker>(model);

    auto fp = inference_test::TupleProviderFactory::Create<int>("A");
    auto bindExpected = fb->BindInputs(fp);
    ASSERT_TRUE((bool)bindExpected);

    auto inputExpected = fb->BindInput<float>("B");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto inputB = inputExpected.value();

    fp->Set<0>(4);
    inputB->Feed(3.0f);

    auto outputExp = fb->BindOutputs<int, float>({"X", "Y"});
    ASSERT_TRUE(outputExp && outputExp.value());
    auto output = outputExp.value();

    std::tuple<int, float> values;
    auto updateExpected = output->UpdateIfChanged(values);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(9, std::get<0>(values));
    ASSERT_EQ(7, std::get<1>(values));
}

TEST(InferenceTestSuite, ProviderThenPipeAlreadyBound) {
    auto fb = std::make_shared<FeatureBroker>();

    auto fp = inference_test::TupleProviderFactory::Create<int>("A");
    auto bindExpected = fb->BindInputs(fp);
    ASSERT_TRUE((bool)bindExpected);

    auto inputExpected = fb->BindInput<int>("A");
    ASSERT_FALSE(inputExpected);
    ASSERT_EQ(feature_errc::already_bound, inputExpected.error());
}

TEST(InferenceTestSuite, PipeThenProviderAlreadyBound) {
    auto fb = std::make_shared<FeatureBroker>();

    auto inputExpected = fb->BindInput<int>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());

    auto fp = inference_test::TupleProviderFactory::Create<int>("A");
    auto bindExpected = fb->BindInputs(fp);
    ASSERT_FALSE(bindExpected);
    ASSERT_EQ(feature_errc::already_bound, bindExpected.error());
}

TEST(InferenceTestSuite, ProviderInvalidInputTypePostModel) {
    auto model = std::make_shared<inference_test::ThreeOutputModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto fp = inference_test::TupleProviderFactory::Create<float>("A");
    auto bindExpected = fb->BindInputs(fp);
    ASSERT_FALSE(bindExpected);
    ASSERT_EQ(feature_errc::type_mismatch, bindExpected.error());
}

TEST(InferenceTestSuite, ProviderInheritance) {
    auto parentFb = std::make_shared<FeatureBroker>();
    auto fp = inference_test::TupleProviderFactory::Create<int>("A");
    auto bindExpected = parentFb->BindInputs(fp);
    ASSERT_TRUE((bool)bindExpected);

    auto model = std::make_shared<inference_test::ThreeOutputModel>();
    auto fbExpected = parentFb->Fork(model);
    ASSERT_TRUE(fbExpected && fbExpected.value());
    auto fb = fbExpected.value();

    int value;
    auto outputExpected = fb->BindOutput<int>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    fp->Set<0>(2);
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7);
}
