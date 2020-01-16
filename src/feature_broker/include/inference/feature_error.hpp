#pragma once

#include <system_error>
#include "tl/expected.hpp"
#include "feature_broker_export.h"

namespace inference {

enum class feature_errc : int {
    ok = 0,
    model_not_found = 1,
    invalid_model,
    name_not_found,
    type_mismatch,
    type_unsupported,
    already_bound,
    not_bound,
    invalid_operation,
    value_update_failure,
    no_model_associated,
    feature_provider_inconsistent,
    circular_structure,
    multiple_waiting,
};

FEATURE_BROKER_EXPORT const std::error_category& feature_error_category() noexcept;

inline const std::error_code make_feature_error(const feature_errc error) noexcept {
    return std::error_code(static_cast<int>(error), feature_error_category());
}

inline const tl::unexpected<std::error_code> make_feature_unexpected(const feature_errc error) noexcept {
    return tl::make_unexpected(make_feature_error(error));
}

inline std::error_condition make_error_condition(inference::feature_errc e) noexcept {
    return std::error_condition(static_cast<int>(e), feature_error_category());
}

inline std::error_code err_feature_ok() { return make_feature_error(feature_errc::ok); }

}  // namespace inference

namespace std {

template <>
struct is_error_condition_enum<inference::feature_errc> : public true_type {};

}  // namespace std
