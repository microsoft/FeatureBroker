// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <system_error>
#include <tl/expected.hpp>

namespace onnx_model {
enum class onnx_errc : int {
    not_built = 1,
    internal_library_error,
    model_load_error,
    unsupported_type,
    unknown_input,
    unknown_output,
    type_mismatch,
    run_error,
    _SENTINEL
};

const std::error_category& onnx_error_category() noexcept;

inline const std::error_code make_onnx_error(const onnx_errc e) noexcept {
    return std::error_code(static_cast<int>(e), onnx_error_category());
}

inline const tl::unexpected<std::error_code> make_onnx_unexpected(const onnx_errc error) noexcept {
    return tl::make_unexpected(make_onnx_error(error));
}

inline std::error_condition make_error_condition(onnx_errc e) noexcept {
    return std::error_condition(static_cast<int>(e), onnx_error_category());
}
}  // namespace inference

namespace std {

template <>
struct is_error_condition_enum<onnx_model::onnx_errc> : public true_type {};

}  // namespace std