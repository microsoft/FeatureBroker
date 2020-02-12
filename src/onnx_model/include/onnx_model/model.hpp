// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <inference/model.hpp>
#include <rt/rt_expected.hpp>
#include <system_error>

#include "onnx_error.hpp"
#include "onnx_model_export.h"

namespace onnx_model {
class Model final : public inference::Model {
   public:
    ONNX_MODEL_EXPORT ~Model() override;
    ONNX_MODEL_EXPORT std::unordered_map<std::string, inference::TypeDescriptor> const& Inputs() const override;
    ONNX_MODEL_EXPORT std::unordered_map<std::string, inference::TypeDescriptor> const& Outputs() const override;
    ONNX_MODEL_EXPORT std::vector<std::string> GetRequirements(std::string const& outputName) const override;

    ONNX_MODEL_EXPORT rt::expected<std::shared_ptr<inference::ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe,
        std::function<void()> outOfBandNotifier) const override;

    ONNX_MODEL_EXPORT static rt::expected<std::shared_ptr<Model>> Load(std::string const& path) noexcept;
    ONNX_MODEL_EXPORT static rt::expected<std::shared_ptr<Model>> LoadFromBuffer(const void* modelData,
                                                                   const size_t modelSize) noexcept;

   private:
    class State;
    const std::shared_ptr<State> m_state;
    std::unordered_map<std::string, inference::TypeDescriptor> m_inputs;
    std::unordered_map<std::string, inference::TypeDescriptor> m_outputs;
    std::vector<std::string> m_deps;

    // Actual types hidden via type erasure.
    Model(void* env, void* session, onnx_errc& errc);

    class UpdaterImpl : public inference::ValueUpdater {
       public:
        class State;
        UpdaterImpl(std::shared_ptr<const Model> parent, std::unique_ptr<State>&& state);
        ~UpdaterImpl();
        std::error_code UpdateOutput();

       private:
        const std::shared_ptr<const Model> m_parent;
        const std::unique_ptr<State> m_state;
    };
};
}  // namespace inference