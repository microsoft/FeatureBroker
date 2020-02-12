// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/direct_input_pipe.hpp>
#include <inference/feature_error.hpp>
#include <inference/model.hpp>
#include <inference/type_descriptor.hpp>
#include <inference/value_updater.hpp>
#include <map>
#include <unordered_map>

using namespace ::inference;

namespace inference_test {
class AddFiveValueUpdater : public inference::ValueUpdater {
   public:
    AddFiveValueUpdater() {}

    AddFiveValueUpdater(std::shared_ptr<IHandle> const& handle, std::shared_ptr<InputPipe> const& pipe)
        : _pipe(std::static_pointer_cast<DirectInputPipe<float>>(pipe)),
          _handle(std::static_pointer_cast<Handle<float>>(handle)) {}

    std::error_code UpdateOutput() override {
        auto value = _handle->Value();
        value += 5.0f;
        _pipe->Feed(value);
        return err_feature_ok();
    }

   private:
    std::shared_ptr<DirectInputPipe<float>> _pipe;
    std::shared_ptr<Handle<float>> _handle;
};

/// <summary>
/// Given a single input 'A' adds five to it and publishes it as the output 'X'.
/// </summary>
class AddFiveModel : public Model {
   public:
    AddFiveModel() {
        _inputs.emplace("A", TypeDescriptor::Create<float>());
        _outputs.emplace("X", TypeDescriptor::Create<float>());
    }

    ~AddFiveModel() {}

    std::unordered_map<std::string, TypeDescriptor> const& Inputs() const override { return _inputs; }

    std::unordered_map<std::string, TypeDescriptor> const& Outputs() const override { return _outputs; }

    std::vector<std::string> GetRequirements(std::string const& outputName) const override {
        std::vector<std::string> requirements = {"A"};
        return requirements;
    }

    rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe,
        std::function<void()> outOfBandNotifier) const override {
        outOfBandNotifier(); // No out of band information, so call and ignore henceforth.
        auto iterPipe = outputToPipe.find("X");
        if (iterPipe == outputToPipe.end()) {
            return make_feature_unexpected(feature_errc::name_not_found);
        }

        auto iterHandle = inputToHandle.find("A");
        if (iterHandle == inputToHandle.end()) {
            return make_feature_unexpected(feature_errc::name_not_found);
        }

        std::shared_ptr<InputPipe> pipe = iterPipe->second;
        std::shared_ptr<IHandle> handle = iterHandle->second;
        auto updater = std::make_shared<AddFiveValueUpdater>(handle, pipe);
        return std::static_pointer_cast<ValueUpdater>(updater);
    }

   private:
    std::unordered_map<std::string, TypeDescriptor> _inputs;
    std::unordered_map<std::string, TypeDescriptor> _outputs;
};

}  // namespace inference_test
