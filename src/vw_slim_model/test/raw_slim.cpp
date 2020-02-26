// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef _WIN32
// There appear to be some includes missing from memory.h that let it compile.
// Also, VW uses throw within memory.h despite VW_NOEXCEPT being set for some
// reason, and this only fails on Linux because the requisite defines needed to
// trigger the compilation of that code are not defined in Windows.
#include <unistd.h>  // So memory.h has access to _SC_PAGE_SIZE.
#define THROW        // So memory.h doesn't fail when THROW is encountered... though this seems unsafe.
#endif

#include <memory>

#include "array_parameters.h"
#include "env.hpp"
#include "example_predict_builder.h"
#include "gtest/gtest.h"
#include "vw_slim_predict.h"
#include "vw_slim_return_codes.h"

namespace raw_models {
#include "data.h"
}

namespace resonance_vw_test {
TEST(Raw, RegressionPredict) {
    // The purpose of this code is to just exercise vw_slim in a more contained and explicit way, corresponding to one
    // of the many subtests in ut_vw.cc.

    auto model_path = test_dir_path + "slimdata/regression_data_3.model";
    auto model_data = all_bytes(model_path);
    ASSERT_EQ(131, model_data.size());

    vw_slim::vw_predict<::sparse_parameters> vw;
    auto code = vw.load(model_data.data(), model_data.size());
    ASSERT_EQ(S_VW_PREDICT_OK, code);

    ::safe_example_predict ex;
    {
        vw_slim::example_predict_builder ex_a(&ex, "a");
        ex_a.push_feature(0, 1.f);
        vw_slim::example_predict_builder ex_b(&ex, "b");
        ex_b.push_feature(2, 2.f);
    }

    float score = 0;
    code = vw.predict(ex, score);
    ASSERT_EQ(S_VW_PREDICT_OK, code);

    // Should be about 0.804214, going by regression_data_3.pred. Give it wiggle room on the last place.
    ASSERT_NEAR(0.804214, score, 1e-6);
}

void sets(::safe_example_predict& ex, int call_type, int modality, int network_type, int platform) {
    vw_slim::example_predict_builder(&ex, "64").push_feature(call_type, 1.f);
    vw_slim::example_predict_builder(&ex, "16").push_feature(modality, 1.f);
    vw_slim::example_predict_builder(&ex, "32").push_feature(network_type, 1.f);
    vw_slim::example_predict_builder(&ex, "48").push_feature(platform, 1.f);
}

TEST(Raw, SkypeJbPredict) {
    vw_slim::vw_predict<::sparse_parameters> vw;

    auto code = vw.load(reinterpret_cast<const char*>(raw_models::cb_data_epsilon_0_skype_jb_model),
                        sizeof(raw_models::cb_data_epsilon_0_skype_jb_model));

    ASSERT_EQ(S_VW_PREDICT_OK, code);

    ::safe_example_predict ex;
    sets(ex, 0, 1, 2, 0);

    const size_t min_delay_actions = 10;
    ::safe_example_predict action_ex[min_delay_actions];
    for (size_t i = 0; i < min_delay_actions; ++i)
        vw_slim::example_predict_builder(&action_ex[i], "80").push_feature(i, 1.f);

    std::vector<float> pdfs;
    std::vector<int> rankings;
    code = vw.predict("eid", ex, action_ex, min_delay_actions, pdfs, rankings);
    ASSERT_EQ(S_VW_PREDICT_OK, code);
}

}  // namespace resonance_vw_test