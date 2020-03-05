#include <vw_remote_model/error.hpp>

namespace resonance_vw {

class VWRemoteErrorCategory final : public std::error_category {
   public:
    const char* name() const noexcept override { return "VW Remote Model"; }

    std::string message(int ev) const noexcept override {
        switch (static_cast<remote_vw_errc>(ev)) {
            case remote_vw_errc::invalid_url:
                return "The specified url is either empty or an invalid format.";
            case remote_vw_errc::invalid_aps_subscription_key:
                return "The specified Azure Personalization Key is missing. Please refer to the Azure portal for this "
                       "information.";
            case remote_vw_errc::rank_response_invalid:
                return "Unable to process the rank response, likely due to an unexpected response format.";
            case remote_vw_errc::rank_request_invalid_event_id:
                return "The event id does not match in the rank response. Please confirm the request contained the expected event id.";
            case remote_vw_errc::rank_request_invalid:
                return "The rank request is invalid, please confirm the inputs and actions are correct.";
            case remote_vw_errc::rank_request_permission_denied:
                return "The rank request failed due to permissions, confirm that the Azure Personalization Key is "
                       "correct.";
            default:
                return "Unknown error code";
        }
    }

    std::error_condition default_error_condition(int c) const noexcept override {
        const remote_vw_errc errc = static_cast<remote_vw_errc>(c);
        switch (errc) {
            case remote_vw_errc::invalid_url:
            case remote_vw_errc::invalid_aps_subscription_key:
            case remote_vw_errc::rank_response_invalid:
            case remote_vw_errc::rank_request_invalid_event_id:
            case remote_vw_errc::rank_request_invalid:
            case remote_vw_errc::rank_request_permission_denied:
                return errc;
            default:
                return std::error_condition(c, *this);
        }
    }
};

const std::error_category& remote_vw_error_category() noexcept {
    static VWRemoteErrorCategory errorCategory;
    return errorCategory;
}

}  // namespace resonance_vw