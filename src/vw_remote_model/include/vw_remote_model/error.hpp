// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <system_error>
#include <tl/expected.hpp>

#include "vw_remote_model_export.h"

namespace resonance_vw {

enum class remote_vw_errc : int {
    invalid_url = 1,
    invalid_aps_subscription_key,
    rank_response_invalid,
    rank_request_invalid_event_id,
    rank_request_invalid = 400,
    rank_request_permission_denied = 401,
    unknown
};

VW_REMOTE_MODEL_EXPORT const std::error_category& remote_vw_error_category() noexcept;

inline const std::error_code make_remote_vw_error(const remote_vw_errc error) noexcept {
    return std::error_code(static_cast<int>(error), remote_vw_error_category());
}

inline const tl::unexpected<std::error_code> make_remote_vw_unexpected(const remote_vw_errc error) noexcept {
    return tl::make_unexpected(make_remote_vw_error(error));
}

inline std::error_condition make_error_condition(remote_vw_errc e) noexcept {
    return std::error_condition(static_cast<int>(e), remote_vw_error_category());
}

}  // namespace vw

namespace std {

template <>
struct is_error_condition_enum<resonance_vw::remote_vw_errc> : public true_type {};

}  // namespace std
