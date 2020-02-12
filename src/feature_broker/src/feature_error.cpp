// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <inference/feature_error.hpp>
#include <string>

namespace inference {

class FeatureErrorCategory final : public std::error_category {
   public:
    const char* name() const noexcept override { return "ValueUpdater"; }

    std::string message(int ev) const noexcept override {
        switch (static_cast<feature_errc>(ev)) {
            case feature_errc::ok:
                return "Success";
            case feature_errc::model_not_found:
                return "Unknown inference task ID";
            case feature_errc::name_not_found:
                return "The input name or output name does not match the "
                       "model.";
            case feature_errc::type_mismatch:
                return "The expected data type does not match.";
            case feature_errc::type_unsupported:
                return "The type is not a supported type.";
            case feature_errc::already_bound:
                return "Specified feature is already bound.";
            case feature_errc::not_bound:
                return "Specified feature is not bound.";
            case feature_errc::invalid_operation:
                return "An invalid operation was performed.";
            case feature_errc::no_model_associated:
                return "No model is associated with this broker.";
            case feature_errc::feature_provider_inconsistent:
                return "The state of the FeatureProvider derived class appears to have mutated. This is disallowed.";
            case feature_errc::circular_structure:
                return "An attempt to introduce a circular structure was detected. This is disallowed.";
            case feature_errc::multiple_waiting:
                return "Multiple waiters appear to be waiting on an output pipe at the same time.";
            default:
                return "Unknown error code";
        }
    }

    std::error_condition default_error_condition(int c) const noexcept override {
        const feature_errc errc = static_cast<feature_errc>(c);
        switch (errc) {
            case feature_errc::ok:
            case feature_errc::model_not_found:
            case feature_errc::name_not_found:
            case feature_errc::type_mismatch:
            case feature_errc::type_unsupported:
            case feature_errc::already_bound:
            case feature_errc::not_bound:
            case feature_errc::invalid_operation:
            case feature_errc::no_model_associated:
            case feature_errc::feature_provider_inconsistent:
            case feature_errc::circular_structure:
                return errc;
            default:
                // Note that feature_errc::value_update_failure will fall through here for pass-through purposes.
                // This is intentional!
                return std::error_condition(c, *this);
        }
    }
};

const std::error_category& feature_error_category() noexcept {
    static FeatureErrorCategory errorCategory;
    return errorCategory;
}

}  // namespace inference
