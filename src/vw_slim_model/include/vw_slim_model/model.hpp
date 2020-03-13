// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/model.hpp>
#include <inference/type_descriptor.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <vw_common/actions.hpp>
#include <vw_common/schema_builder.hpp>
#include <vw_slim_model/output_task.hpp>

#include "vw_slim_model_export.h"

namespace resonance_vw {

class Model : public inference::Model {
   public:
    VW_SLIM_MODEL_EXPORT static rt::expected<std::shared_ptr<Model>> Load(SchemaBuilder const& schemaBuilder,
                                                                          std::shared_ptr<OutputTask> task,
                                                                          const void* modelData, size_t modelSize);

    VW_SLIM_MODEL_EXPORT ~Model() override;
    VW_SLIM_MODEL_EXPORT std::unordered_map<std::string, inference::TypeDescriptor> const& Inputs() const override;
    VW_SLIM_MODEL_EXPORT std::unordered_map<std::string, inference::TypeDescriptor> const& Outputs() const override;
    VW_SLIM_MODEL_EXPORT std::vector<std::string> GetRequirements(std::string const& outputName) const override;

    VW_SLIM_MODEL_EXPORT rt::expected<std::shared_ptr<inference::ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe,
        std::function<void()> outOfBandNotifier) const override;

    class State;

   private:
    std::shared_ptr<State> m_state;

    Model(SchemaBuilder const& schemaBuilder, std::shared_ptr<OutputTask> task, std::shared_ptr<void> vwPredict);

    std::unordered_map<std::string, inference::TypeDescriptor> m_inputs;
    std::vector<std::string> m_input_names;
};
}  // namespace resonance_vw
