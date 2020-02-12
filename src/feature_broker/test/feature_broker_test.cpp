// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <inference/feature_broker.hpp>
#include <inference/synchronous_feature_broker.hpp>
#include <memory>

#include "add_five_model.hpp"
#include "add_model.hpp"
#include "error_model.hpp"
#include "gtest/gtest.h"

using namespace ::inference;

TEST(InferenceTestSuite, FeatureBrokerCreation) { FeatureBroker fb; }

TEST(InferenceTestSuite, FeatureBrokerSingleInputAndOutput) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    float value;
    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    input->Feed(2.0f);
    ASSERT_TRUE(output->Changed());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);
}

TEST(InferenceTestSuite, SynchronousFeatureBrokerSingleInputAndOutput) {
    // Similar to the feature broker above, in fact usage should appear identical, but with no support for async
    // operations, inheritance, hierarchy, and so forth.
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<SynchronousFeatureBroker>(model);

    float value;

    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    input->Feed(2.0f);
    ASSERT_TRUE(output->Changed());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);
}

TEST(InferenceTestSuite, FeatureBrokerSingleInputAndOutputInspectInputMap) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    float value;
    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto inputMap = output->Inputs();
    auto outputInputMapFind = inputMap.find("A");
    ASSERT_FALSE(inputMap.end() == outputInputMapFind);
    auto handleExpected = TryCast<float>(outputInputMapFind->second);
    ASSERT_TRUE(handleExpected && handleExpected.value());
    auto handle = handleExpected.value();

    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    input->Feed(2.0f);
    ASSERT_TRUE(output->Changed());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);
    ASSERT_EQ(2.0f, handle->Value());

    input->Feed(3.5f);
    ASSERT_EQ(2.0f, handle->Value());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(3.5f, handle->Value());
}

TEST(InferenceTestSuite, SynchronousFeatureBrokerMultiInputSingleOutput) {
    auto model = std::make_shared<inference_test::AddModel>();
    auto fb = std::make_shared<SynchronousFeatureBroker>(model);

    float value;

    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto inputA = inputExpected.value();

    inputExpected = fb->BindInput<float>("B");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto inputB = inputExpected.value();

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    // No inputs fed yet. Should not be changed.
    ASSERT_FALSE(output->Changed());
    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    // Only one of the required inputs fed. Should not be changed.
    inputA->Feed(2);
    ASSERT_FALSE(output->Changed());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    // Still only one of the required inputs fed. Should not be changed.
    inputA->Feed(1);
    ASSERT_FALSE(output->Changed());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    // Finally feed B. Now should be changed, but marked as unchanged after the value is fetched.
    inputB->Feed(-3);
    ASSERT_TRUE(output->Changed());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(-2, value);
    ASSERT_FALSE(output->Changed());

    // Now that both the inputs have been fetched, changing
    // a *single* input value should now trigger the change.
    inputA->Feed(2);
    ASSERT_TRUE(output->Changed());
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(-1, value);
    ASSERT_FALSE(output->Changed());
}

TEST(InferenceTestSuite, HandleTryCast) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto inputMap = output->Inputs();
    auto outputInputMapFind = inputMap.find("A");
    ASSERT_FALSE(inputMap.end() == outputInputMapFind);
    auto untypedHandle = outputInputMapFind->second;
    auto handleExpected = TryCast<float>(untypedHandle);
    ASSERT_TRUE(handleExpected && handleExpected.value());

    auto handleExpected2 = TryCast<int>(untypedHandle);
    ASSERT_FALSE(handleExpected2);
    ASSERT_EQ(feature_errc::type_mismatch, handleExpected2.error());
}

TEST(InferenceTestSuite, FeatureBrokerSingleInputAndOutputErrorModel) {
    auto model = std::make_shared<inference_test::ErrorIfThreeModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    float value;
    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    input->Feed(2.0f);
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);

    // The ErrorIfThreeModel fails if the input is 3, so we expect failure here.
    input->Feed(3.0f);
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_FALSE(updateExpected);

    input->Feed(4.0f);
    updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 9.0);
}

TEST(InferenceTestSuite, FeatureBrokerInputFeedBeforeOuputBinding) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    float value = 2.0f;

    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();
    input->Feed(2.0f);

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    auto updateExpected = output->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);
}

TEST(InferenceTestSuite, FeatureBrokerMultipleInputs) {
    auto fb = std::make_shared<FeatureBroker>();

    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto inputExpected2 = fb->BindInput<int>("B");
    ASSERT_TRUE(inputExpected2 && inputExpected2.value());
    auto input2 = inputExpected2.value();

    auto inputExpected3 = fb->BindInput<std::string>("C");
    ASSERT_TRUE(inputExpected3 && inputExpected3.value());
    auto input3 = inputExpected3.value();
}

TEST(InferenceTestSuite, FeatureBrokerInvalidInputTypePostModel) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto inputExpected = fb->BindInput<int>("A");
    ASSERT_FALSE(inputExpected);
    ASSERT_EQ(feature_errc::type_mismatch, inputExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerInputAlreadyBound) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());

    auto inputExpected2 = fb->BindInput<float>("A");
    ASSERT_FALSE(inputExpected2);
    ASSERT_EQ(feature_errc::already_bound, inputExpected2.error());
}

TEST(InferenceTestSuite, FeatureBrokerInvalidOutputName) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto outputExpected = fb->BindOutput<float>("Y");
    ASSERT_FALSE(outputExpected);
    ASSERT_EQ(feature_errc::name_not_found, outputExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerMissingInput) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_FALSE(outputExpected);
    ASSERT_EQ(feature_errc::not_bound, outputExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerMissingModel) {
    auto fb = std::make_shared<FeatureBroker>();

    auto outputExpected = fb->BindOutput<float>("X");
    ASSERT_FALSE(outputExpected);
    ASSERT_EQ(feature_errc::no_model_associated, outputExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerHeirarchyWithLateBoundInputsAndAssociatedModels) {
    auto broker = std::make_shared<FeatureBroker>();
    auto model = std::make_shared<inference_test::AddFiveModel>();

    auto subBroker1Expected = broker->Fork(model);
    ASSERT_TRUE(subBroker1Expected && subBroker1Expected.value());
    auto subBroker1 = subBroker1Expected.value();
    auto inputExpected = broker->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();
    auto subBroker2Expected = broker->Fork(model);
    ASSERT_TRUE(subBroker2Expected && subBroker2Expected.value());
    auto subBroker2 = subBroker2Expected.value();

    auto outputExpected = subBroker1->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output1 = outputExpected.value();

    outputExpected = subBroker2->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output2 = outputExpected.value();

    float value;
    ASSERT_FALSE(output1->Changed());
    ASSERT_FALSE(output2->Changed());
    input->Feed(2.f);
    ASSERT_TRUE(output1->Changed());
    auto updateExpected = output1->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);
    updateExpected = output1->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());
    value = 0.f;
    updateExpected = output2->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(value, 7.0);
}

TEST(InferenceTestSuite, FeatureBrokerHeirarchyTypeMismatch) {
    auto fb = std::make_shared<FeatureBroker>();
    auto model = std::make_shared<inference_test::AddFiveModel>();

    auto inputExpected = fb->BindInput<int>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    // AddFiveModel is a model that wants "A" as a float, so it should complain
    // about the type mismatch from the prior binding of "A" as int.
    auto forkedExpected = fb->Fork(model);
    ASSERT_FALSE(forkedExpected);
    ASSERT_EQ(feature_errc::type_mismatch, forkedExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerSetParentTypeBecomesOK) {
    auto model1 = std::make_shared<inference_test::AddModel>();
    auto model2 = std::make_shared<inference_test::AddFiveModel>();

    auto fb1 = std::make_shared<FeatureBroker>(model1);
    auto fb2 = std::make_shared<FeatureBroker>(model2);

    auto fb3 = fb1->Fork().value_or(nullptr);
    ASSERT_NE(nullptr, fb3);
    auto inputA = fb3->BindInput<float>("A").value_or(nullptr);
    ASSERT_NE(nullptr, inputA);
    // It should first fail, since B is an input of type float.
    auto inputBExpected = fb3->BindInput<int>("B");
    ASSERT_FALSE(inputBExpected);
    ASSERT_EQ(feature_errc::type_mismatch, inputBExpected.error());

    // After this, however, we should be fine, since the input B is no longer consumed.
    ASSERT_TRUE((bool)fb3->SetParent(fb2));
    auto inputB = fb3->BindInput<int>("B").value_or(nullptr);
    ASSERT_NE(nullptr, inputB);
}

TEST(InferenceTestSuite, FeatureBrokerSetParentChangesModel) {
    // In this test, we use the change of a parent to change the model.
    auto model1 = std::make_shared<inference_test::AddModel>();
    auto model2 = std::make_shared<inference_test::AddFiveModel>();

    auto fb1 = std::make_shared<FeatureBroker>(model1);
    auto fb2 = std::make_shared<FeatureBroker>(model2);

    auto fb3 = fb1->Fork().value_or(nullptr);
    ASSERT_NE(nullptr, fb3);
    auto inputA = fb3->BindInput<float>("A").value_or(nullptr);
    ASSERT_NE(nullptr, inputA);
    auto inputB = fb3->BindInput<float>("B").value_or(nullptr);
    ASSERT_NE(nullptr, inputB);
    inputA->Feed(1);
    inputB->Feed(2);

    // This first binding of the output will be against the AddModel.
    auto output1 = fb3->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, output1);
    float value;
    auto updateExpected = output1->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(3, value);

    // Reassign the parent, this one associated with the AddFiveModel.
    ASSERT_TRUE((bool)fb3->SetParent(fb2));
    // This second binding of the output will be against the AddFiveModel.
    auto output2 = fb3->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, output2);
    updateExpected = output2->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(6, value);
    // The original output, still against AddModel, should not report a change.
    updateExpected = output1->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    // Now feed a new value to B. This should change output1 but not output2.
    inputB->Feed(3);
    updateExpected = output1->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(4, value);
    updateExpected = output2->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && !updateExpected.value());
}

TEST(InferenceTestSuite, FeatureBrokerSetParentChangesInputs) {
    // In this test, we use the change of a parent to change the inputs.

    auto fb1 = std::make_shared<FeatureBroker>();
    auto fb2 = std::make_shared<FeatureBroker>();

    auto input1 = fb1->BindInput<float>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input1);
    auto input2 = fb2->BindInput<float>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input2);

    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb3 = fb1->Fork(model).value_or(nullptr);

    // Bind outputs twice, first against the first parent, then against the second.
    auto output1 = fb3->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, output1);
    ASSERT_TRUE((bool)fb3->SetParent(fb2));
    auto output2 = fb3->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, output2);

    input1->Feed(1);
    input2->Feed(2);

    float value;
    auto updateExpected = output1->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(6, value);
    updateExpected = output2->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(7, value);
}

TEST(InferenceTestSuite, FeatureBrokerSetParentNullChangesInputs) {
    // Similar to the above FeatureBrokerSetParentChangesInputs, except that we set the parent to null to "unparent" it,
    // then bind again on the child which,

    auto fb1 = std::make_shared<FeatureBroker>();

    auto input1 = fb1->BindInput<float>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input1);

    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb2 = fb1->Fork(model).value_or(nullptr);

    // Bind outputs twice, first against the first parent, then against the second.
    auto output1 = fb2->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, output1);

    ASSERT_TRUE((bool)fb2->SetParent(nullptr));
    // Since we've unparented fb2, this should be fine.
    auto input2 = fb2->BindInput<float>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input2);

    auto output2 = fb2->BindOutput<float>("X").value_or(nullptr);
    ASSERT_NE(nullptr, output2);

    input1->Feed(1);
    input2->Feed(2);

    float value;
    auto updateExpected = output1->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(6, value);
    updateExpected = output2->UpdateIfChanged(value);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(7, value);
}

TEST(InferenceTestSuite, FeatureBrokerSetParentAlreadyBound) {
    auto fb1 = std::make_shared<FeatureBroker>();
    auto fb2 = std::make_shared<FeatureBroker>();
    auto input1 = fb1->BindInput<int>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input1);
    auto input2 = fb2->BindInput<int>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input2);
    auto setParentExpected = fb2->SetParent(fb1);
    ASSERT_FALSE(setParentExpected);
    ASSERT_EQ(feature_errc::already_bound, setParentExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerSetParentChildModelTypeMismatch) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb1 = std::make_shared<FeatureBroker>();
    auto fb2 = std::make_shared<FeatureBroker>(model);
    auto input = fb1->BindInput<int>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input);
    // This should fail because the parent's input binding conflicts with its new child's model.
    auto setParentExpected = fb2->SetParent(fb1);
    ASSERT_FALSE(setParentExpected);
    ASSERT_EQ(feature_errc::type_mismatch, setParentExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerSetParentParentModelTypeMismatch) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb1 = std::make_shared<FeatureBroker>(model);
    auto fb2 = std::make_shared<FeatureBroker>();
    auto input = fb2->BindInput<int>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input);
    // This should fail because the parent's model conflicts with its new child's input binding.
    auto setParentExpected = fb2->SetParent(fb1);
    ASSERT_FALSE(setParentExpected);
    ASSERT_EQ(feature_errc::type_mismatch, setParentExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerSetParentModelTypeMismatchLate) {
    // This test differs from the one above insofar that because the hierarchy is *three* deep, and we do the resetting
    // of a parent on the middle element (whereas the conflict happens in the last element), we can only detect the
    // error on UpdateIfChanged.
    auto fb1 = std::make_shared<FeatureBroker>();
    auto fb2 = fb1->Fork().value_or(nullptr);
    ASSERT_NE(nullptr, fb2);
    auto fb3 = fb2->Fork().value_or(nullptr);

    auto input = fb3->BindInput<int>("A").value_or(nullptr);
    ASSERT_NE(nullptr, input);

    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fbNew1 = std::make_shared<FeatureBroker>(model);

    // This has no mechanism to fail, since fb2 doesn't know its children.
    ASSERT_TRUE((bool)fb2->SetParent(fbNew1));

    // This should fail, since "A" is incompatible.
    auto expectedOutput = fb3->BindOutput<float>("X");
    ASSERT_FALSE(expectedOutput);
    ASSERT_EQ(feature_errc::type_mismatch, expectedOutput.error());
}

TEST(InferenceTestSuite, FeatureBrokerSetParentCyclicStructure) {
    auto fb1 = std::make_shared<FeatureBroker>();
    auto fb2 = fb1->Fork().value_or(nullptr);
    ASSERT_NE(nullptr, fb2);
    auto fb3 = fb2->Fork().value_or(nullptr);
    ASSERT_NE(nullptr, fb3);
    auto setParentExpected = fb1->SetParent(fb3);
    ASSERT_FALSE(setParentExpected);
    ASSERT_EQ(feature_errc::circular_structure, setParentExpected.error());
}

TEST(InferenceTestSuite, FeatureBrokerWithBoundOutputOutOfScope) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto fb = std::make_shared<FeatureBroker>(model);
    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    {
        auto outputExpected = fb->BindOutput<float>("X");
        ASSERT_TRUE(outputExpected && outputExpected.value());
        input->Feed(2.f);
    }

    input->Feed(2.f);
}

TEST(InferenceTestSuite, FeatureBrokerInheritanceInput) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto parentFb = std::make_shared<FeatureBroker>();
    auto inputExpected = parentFb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto childFbExpected = parentFb->Fork(model);
    ASSERT_TRUE(childFbExpected && childFbExpected.value());
    auto childFb = childFbExpected.value();
    input->Feed(2.f);

    auto outputExpected = childFb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    float value;
    outputExpected.value()->UpdateIfChanged(value);
    ASSERT_TRUE(outputExpected && outputExpected.value());
    ASSERT_EQ(7, value);
}

TEST(InferenceTestSuite, FeatureBrokerInheritanceInputAlreadyBound) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto parentFb = std::make_shared<FeatureBroker>();

    auto inputExpected = parentFb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto childFbExpected = parentFb->Fork(model);
    ASSERT_TRUE(childFbExpected && childFbExpected.value());
    auto childFb = childFbExpected.value();
    auto inputExpected2 = childFb->BindInput<float>("A");
    ASSERT_FALSE(inputExpected2);
    ASSERT_EQ(feature_errc::already_bound, inputExpected2.error());
}

TEST(InferenceTestSuite, FeatureBrokerInheritanceParentAssociatedWithModel) {
    auto model = std::make_shared<inference_test::AddFiveModel>();
    auto parentFb = std::make_shared<FeatureBroker>(model);
    auto inputExpected = parentFb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    auto outputExpected = parentFb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    // If child is created from parent and the parent contains a model, the child will use the parents model
    auto childFbExpected = parentFb->Fork();
    ASSERT_TRUE(childFbExpected && childFbExpected.value());
    auto childFb = childFbExpected.value();
    outputExpected = childFb->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output2 = outputExpected.value();
}

TEST(InferenceTestSuite, FeatureBrokerInheritanceModelWithMultipleInputs) {
    auto model = std::make_shared<inference_test::AddModel>();
    auto fb = std::make_shared<FeatureBroker>();

    // bind "A" at the parent feature broker
    auto inputExpected = fb->BindInput<float>("A");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input = inputExpected.value();

    // bind "B" at the child feature broker
    auto fb2Expected = fb->Fork(model);
    ASSERT_TRUE(fb2Expected && fb2Expected.value());
    auto fb2 = fb2Expected.value();
    inputExpected = fb2->BindInput<float>("B");
    ASSERT_TRUE(inputExpected && inputExpected.value());
    auto input2 = inputExpected.value();

    auto outputExpected = fb2->BindOutput<float>("X");
    ASSERT_TRUE(outputExpected && outputExpected.value());
    auto output = outputExpected.value();

    input->Feed(1.f);
    input2->Feed(2.f);
    float outputVal;

    auto updateExpected = output->UpdateIfChanged(outputVal);
    ASSERT_TRUE(updateExpected && updateExpected.value());
    ASSERT_EQ(3.f, outputVal);
}
