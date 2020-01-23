#pragma once

#include <memory>
#include <string>
#include <system_error>
#include <tl/expected.hpp>
#include <unordered_set>

#include "vw_slim_model_export.h"

namespace vw_slim_model {
/**
 * @brief Creates a structure to describe the input schema for a model.
 *
 * Unlike, for instance, a TensorFlow or ONNX model, the schema of the expected inputs to a model is not baked into part
 * of the model binary content. Given VW's trick of reducing such extra non-feature-value information like namespaces
 * and feature names and indices into a simple hash, this makes perfect sense. Nonetheless as a practical matter, this
 * flexibility of literally anything being an acceptable feature name led to a lot of desparities between the training
 * and the deployment of models, which we resolve with a schema.
 */
class SchemaBuilder {
   public:
    VW_SLIM_MODEL_EXPORT SchemaBuilder();

    using Expected = tl::expected<void, std::error_code>;

    /**
     * @brief Adds a scalar float feature to the schema.
     *
     * As an example, if this is specified with namespace `foo` and feature name `bar`, and the corresponding input pipe
     * receives a value of 5, this would be akin to the VW example `|foo bar:5`.
     *
     * @param inputName The name of the input.
     * @param featureName The corresponding name of the feature in VW.
     * @param ns The corresponding name of the namespace in VW.
     * @return VW_SLIM_MODEL_EXPORT AddFeature
     */
    VW_SLIM_MODEL_EXPORT Expected AddFloatFeature(std::string const& inputName, std::string const& featureName,
                                                  std::string const& ns = "");

    /**
     * @brief Adds a scalar float feature to the schema.
     *
     * As an example, if this is specified with namespace `foo` and feature index `3`, and the corresponding input pipe
     * receives a value of 5, this would be akin to the VW example `|foo 3:5`.
     *
     * @param inputName The name of the input.
     * @param offset The offset index in VW.
     * @param ns The corresponding name of the namespace in VW.
     * @return VW_SLIM_MODEL_EXPORT AddFeature
     */
    VW_SLIM_MODEL_EXPORT Expected AddFloatFeature(std::string const& inputName, std::size_t offset,
                                                  std::string const& ns = "");

    VW_SLIM_MODEL_EXPORT Expected AddFloatVectorFeature(std::string const& inputName, std::size_t offset = 0,
                                                        std::string const& ns = "");

    VW_SLIM_MODEL_EXPORT Expected AddStringFeature(std::string const& inputName, std::string const& ns = "");

    VW_SLIM_MODEL_EXPORT Expected AddStringVectorFeature(std::string const& inputName, std::string const& ns = "");

   private:
    friend class Model;
    std::shared_ptr<void> m_schema;
    std::unordered_set<std::string> m_input_names;
};
}  // namespace vw_slim_model