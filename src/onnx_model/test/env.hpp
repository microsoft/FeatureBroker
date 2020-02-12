#pragma once

#include <string>
#include <vector>

namespace onnx_model_test {
std::vector<char> all_bytes(std::string path);

extern std::string test_dir_path;
}  // namespace onnx_model_test