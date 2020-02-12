#include "onnx_model/onnx_error.hpp"
#include <string>
#include <system_error>

namespace onnx_model {
// Local error code implementation. Only used in this file.
class OnnxModelErrorCategory final : public std::error_category {
   public:
    const char* name() const noexcept override { return "OnnxModel"; }

    std::string message(int ev) const noexcept override {
        switch (static_cast<onnx_errc>(ev)) {
            case onnx_errc::not_built:
                return "This appears to not have been built with ONNX.";
            case onnx_errc::internal_library_error:
                return "Could not create internal ONNX structure.";
            case onnx_errc::model_load_error:
                return "Could not load model.";
            case onnx_errc::unsupported_type:
                return "Unsupported type.";
            case onnx_errc::unknown_input:
                return "A name was provided as an input that is unknown.";
            case onnx_errc::unknown_output:
                return "A name was provided as an output that is unknown.";
            case onnx_errc::type_mismatch:
                return "Mismatch on expected types.";
            case onnx_errc::run_error:
                return "Error happened during ONNX inference.";
            default:
                return "Unknown error code.";
        }
    }

    std::error_condition default_error_condition(int c) const noexcept override {
        const onnx_errc errc = static_cast<onnx_errc>(c);
        if (onnx_errc::not_built <= errc && errc < onnx_errc::_SENTINEL) return errc;
        return std::error_condition(c, *this);
    }
};

const std::error_category& onnx_error_category() noexcept {
    static onnx_model::OnnxModelErrorCategory error_category;
    return error_category;
}
}  // namespace inference
