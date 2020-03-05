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

#include "vw_remote_model_export.h"

namespace resonance_vw {

class IRecommenderClient;

class RemoteModel : public inference::Model {
   public:
    VW_REMOTE_MODEL_EXPORT static rt::expected<std::shared_ptr<Model>> Load(SchemaBuilder const& schemaBuilder,
                                                                            std::shared_ptr<Actions> actions,
                                                                            std::shared_ptr<IRecommenderClient> client);

    VW_REMOTE_MODEL_EXPORT ~RemoteModel() override;
    VW_REMOTE_MODEL_EXPORT std::unordered_map<std::string, inference::TypeDescriptor> const& Inputs() const override;
    VW_REMOTE_MODEL_EXPORT std::unordered_map<std::string, inference::TypeDescriptor> const& Outputs() const override;
    VW_REMOTE_MODEL_EXPORT std::vector<std::string> GetRequirements(std::string const& outputName) const override;

    VW_REMOTE_MODEL_EXPORT rt::expected<std::shared_ptr<inference::ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe,
        std::function<void()> outOfBandNotifier) const override;

   private:
    RemoteModel(SchemaBuilder const& schemaBuilder, std::shared_ptr<Actions> actions,
                std::shared_ptr<IRecommenderClient> client);

    const std::shared_ptr<void> m_schema;
    const std::shared_ptr<Actions> m_actions;
    std::unordered_map<std::string, inference::TypeDescriptor> m_inputs;
    std::unordered_map<std::string, inference::TypeDescriptor> m_outputs;
    std::vector<std::string> m_input_names;
    std::shared_ptr<IRecommenderClient> m_client;
};
}  // namespace resonance_vw
