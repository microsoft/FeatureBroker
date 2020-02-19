// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <system_error>
#include <tl/expected.hpp>
#include "vw_slim_model_export.h"

namespace vw_slim_model {

enum class vw_errc : int {
    duplicate_input_name = 1,
    load_failure,
    invalid_actions,
};

VW_SLIM_MODEL_EXPORT const std::error_category& vw_error_category() noexcept;

inline const std::error_code make_vw_error(const vw_errc error) noexcept {
    return std::error_code(static_cast<int>(error), vw_error_category());
}

inline const tl::unexpected<std::error_code> make_vw_unexpected(const vw_errc error) noexcept {
    return tl::make_unexpected(make_vw_error(error));
}

inline std::error_condition make_error_condition(vw_errc e) noexcept {
    return std::error_condition(static_cast<int>(e), vw_error_category());
}

}  // namespace vw_slim_model

namespace std {

template <>
struct is_error_condition_enum<vw_slim_model::vw_errc> : public true_type {};

}  // namespace std
