// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <string>
#include <vw_slim_model/vw_error.hpp>

namespace vw_slim_model {

class VWErrorCategory final : public std::error_category {
   public:
    const char* name() const noexcept override { return "VW Slim Model"; }

    std::string message(int ev) const noexcept override {
        switch (static_cast<vw_errc>(ev)) {
            case vw_errc::duplicate_input_name:
                return "The input name was used multiple times.";
            case vw_errc::load_failure:
                return "Failure to load a VW predict object.";
            default:
                return "Unknown error code";
        }
    }

    std::error_condition default_error_condition(int c) const noexcept override {
        const vw_errc errc = static_cast<vw_errc>(c);
        switch (errc) {
            case vw_errc::duplicate_input_name:
            case vw_errc::load_failure:
                return errc;
            default:
                return std::error_condition(c, *this);
        }
    }
};

const std::error_category& vw_error_category() noexcept {
    static VWErrorCategory errorCategory;
    return errorCategory;
}

}  // namespace vw_slim_model
