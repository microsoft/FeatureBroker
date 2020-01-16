#include <inference/feature_broker.hpp>
#include <memory>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "three_output_model.hpp"

using namespace ::inference;

TEST(InferenceTestSuite, ThreeOutputModelSingleConsumption) {
    auto model = std::make_shared<inference_test::ThreeOutputModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto inputAExp = fb->BindInput<int>("A");
    ASSERT_TRUE((bool)inputAExp);
    auto inputA = inputAExp.value();

    auto outputXExp = fb->BindOutput<int>("X");
    ASSERT_TRUE((bool)outputXExp);
    auto outputX = outputXExp.value();

    int value;
    auto update = outputX->UpdateIfChanged(value);
    ASSERT_TRUE(update && !update.value());

    inputA->Feed(2);
    update = outputX->UpdateIfChanged(value);
    ASSERT_TRUE(update && update.value());
    ASSERT_EQ(value, 7);
}

TEST(InferenceTestSuite, ThreeOutputModelTuple) {
    auto model = std::make_shared<inference_test::ThreeOutputModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto inputAExp = fb->BindInput<int>("A");
    ASSERT_TRUE((bool)inputAExp);
    auto inputA = inputAExp.value();

    auto outputExp = fb->BindOutputs<int, std::string>({"X", "Z"});
    ASSERT_TRUE(outputExp && outputExp.value());
    auto output = outputExp.value();

    std::tuple<int, std::string> value;
    auto update = output->UpdateIfChanged(value);
    ASSERT_TRUE(update && !update.value());

    inputA->Feed(2);
    update = output->UpdateIfChanged(value);
    ASSERT_TRUE(update && update.value());
    ASSERT_EQ(std::get<0>(value), 7);
    ASSERT_EQ(std::get<1>(value), "2");
}

TEST(InferenceTestSuite, ThreeOutputModelInputUnbound) {
    auto model = std::make_shared<inference_test::ThreeOutputModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto input = fb->BindInput<int>("A");
    ASSERT_TRUE(input && input.value());

    // Y depends on B, but B is not bound.
    auto output = fb->BindOutputs<int, float>({"X", "Y"});
    ASSERT_FALSE(output);
    ASSERT_EQ(feature_errc::not_bound, output.error());
}

TEST(InferenceTestSuite, ThreeOutputModelTypeMismatch) {
    auto model = std::make_shared<inference_test::ThreeOutputModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto input = fb->BindInput<int>("A");
    ASSERT_TRUE(input && input.value());

    // Z should be std::string, but the argument was listed as a float.
    auto output = fb->BindOutputs<int, float>({"X", "Z"});
    ASSERT_FALSE(output);
    ASSERT_EQ(feature_errc::type_mismatch, output.error());
}

TEST(InferenceTestSuite, ThreeOutputModelNamesSizeUnmatched) {
    auto model = std::make_shared<inference_test::ThreeOutputModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto input = fb->BindInput<int>("A");
    ASSERT_TRUE(input && input.value());

    // Three names specified, but should be only two.
    auto output = fb->BindOutputs<int, std::string>({"X", "Z", "Y"});
    // Is a more specific error message appropriate here.
    ASSERT_FALSE(output);
    ASSERT_EQ(feature_errc::invalid_operation, output.error());
}
