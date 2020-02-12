#include <string>

#include "env.hpp"
#include "gtest/gtest.h"

using namespace onnx_model_test;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Attempt to infer the path to the exe from the first argument.
    assert(argc > 0);
    std::string exe_path(argv[0]);
    auto sep_position = exe_path.find_last_of("/\\");
    if (sep_position != std::string::npos)
        test_dir_path = exe_path.substr(0, sep_position + 1);
    else
        test_dir_path = "";

    return RUN_ALL_TESTS();
}
