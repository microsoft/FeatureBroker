// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "onnx_model/model.hpp"

#include <gtest/gtest.h>

#include <codecvt>
#include <inference/feature_broker.hpp>
#include <locale>
#include <memory>
#include <string>

#include "env.hpp"

using namespace inference;
namespace onnx_model_test {


TEST(FeatureBrokerOnnx, CheckModelOutputsMatrixMultiply) {
    auto path = test_dir_path + "matmul.onnx";
    auto modelExpected = onnx_model::Model::Load(path);
    ASSERT_TRUE(modelExpected && modelExpected.value());
    auto model = modelExpected.value();
    auto inputs = model->Inputs();
    ASSERT_EQ(2, inputs.size());

    auto found = inputs.find("A:0");
    ASSERT_NE(inputs.end(), found);
    auto typeDesc = TypeDescriptor::Create<Tensor<double>>();
    ASSERT_EQ(typeDesc, found->second);

    found = inputs.find("B:0");
    ASSERT_NE(inputs.end(), found);
    ASSERT_EQ(typeDesc, found->second);

    auto outputs = model->Outputs();

    found = outputs.find("C:0");
    ASSERT_NE(outputs.end(), found);
    ASSERT_EQ(typeDesc, found->second);
}

TEST(FeatureBrokerOnnx, RunMatrixMultiply) {
    auto path = test_dir_path + "matmul.onnx";
    auto modelExpected = onnx_model::Model::Load(path);
    ASSERT_TRUE((bool)modelExpected);
    auto model = modelExpected.value();

    FeatureBroker fb(model);
    auto inputA = fb.BindInput<Tensor<double>>("A:0").value_or(nullptr);
    ASSERT_NE(nullptr, inputA);
    auto inputB = fb.BindInput<Tensor<double>>("B:0").value_or(nullptr);
    ASSERT_NE(nullptr, inputB);
    auto output = fb.BindOutput<Tensor<double>>("C:0").value_or(nullptr);
    ASSERT_NE(nullptr, output);

    auto adata = std::shared_ptr<double>(new double[6]{1, 2, 3, 4, 5, 6});
    auto bdata = std::shared_ptr<double>(new double[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    //       / 1 2 3 \ / 1  2  3  4 \   / 38 44  50  56 \
    // A B = \ 4 5 6 / | 5  6  7  8 | = \ 83 98 113 128 /
    //                 \ 9 10 11 12 /
    Tensor<double> a(adata, {2, 3});
    Tensor<double> b(bdata, {3, 4});
    inputA->Feed(a);
    inputB->Feed(b);

    Tensor<double> c;
    auto updateExpected = output->UpdateIfChanged(c);
    ASSERT_TRUE(updateExpected && updateExpected.value());

    ASSERT_EQ(2, c.Dimensions().size());
    ASSERT_EQ(2, c.Dimensions().at(0));
    ASSERT_EQ(4, c.Dimensions().at(1));

    const double* cdata = c.Data();
    ASSERT_EQ(38, cdata[0]);
    ASSERT_EQ(44, cdata[1]);
    ASSERT_EQ(50, cdata[2]);
    ASSERT_EQ(56, cdata[3]);
    ASSERT_EQ(83, cdata[4]);
    ASSERT_EQ(98, cdata[5]);
    ASSERT_EQ(113, cdata[6]);
    ASSERT_EQ(128, cdata[7]);

    // Don't change...
    updateExpected = output->UpdateIfChanged(c);
    ASSERT_TRUE(updateExpected && !updateExpected.value());

    b = Tensor<double>(bdata, {3, 1});
    //       / 1 2 3 \ / 1 \   / 14 \
    // A B = \ 4 5 6 / | 2 | = \ 32 /
    //                 \ 3 /
    inputB->Feed(b);
    Tensor<double> d;
    // Now it should be changed again.
    updateExpected = output->UpdateIfChanged(d);
    ASSERT_TRUE(updateExpected && updateExpected.value());

    ASSERT_EQ(2, d.Dimensions().size());
    ASSERT_EQ(2, d.Dimensions().at(0));
    ASSERT_EQ(1, d.Dimensions().at(1));

    const double* ddata = d.Data();
    ASSERT_EQ(14, ddata[0]);
    ASSERT_EQ(32, ddata[1]);

    // Make sure that the C tensor hasn't changed.
    cdata = c.Data();
    ASSERT_EQ(38, cdata[0]);
    ASSERT_EQ(44, cdata[1]);
    ASSERT_EQ(50, cdata[2]);
    ASSERT_EQ(56, cdata[3]);
    ASSERT_EQ(83, cdata[4]);
    ASSERT_EQ(98, cdata[5]);
    ASSERT_EQ(113, cdata[6]);
    ASSERT_EQ(128, cdata[7]);
}
}  // namespace onnx_model_test
