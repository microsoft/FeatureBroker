// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/direct_input_pipe.hpp>
#include <inference/feature_error.hpp>
#include <inference/model.hpp>
#include <inference/type_descriptor.hpp>
#include <inference/value_updater.hpp>
#include <map>
#include <memory>
#include <system_error>
#include <unordered_map>

using namespace ::inference;

namespace inference_test {
/// Given a single input 'A' adds five to it and publishes it as the output 'X', unless the input happens to be 3, in
/// which case it yields an error.
class ErrorIfThreeModel : public Model {
   private:
    class ValueUpdaterImpl : public inference::ValueUpdater {
       public:
        ValueUpdaterImpl(std::shared_ptr<IHandle> const &handle, std::shared_ptr<InputPipe> const &pipe)
            : _pipe(std::static_pointer_cast<DirectInputPipe<float>>(pipe)),
              _handle(std::static_pointer_cast<Handle<float>>(handle)) {}

        std::error_code UpdateOutput() override {
            if (!_handle || !_pipe) return err_feature_ok();
            // In case the input is 3, just throw one of the standard errors.
            if (_handle->Value() == 3) return std::make_error_code(std::errc::invalid_argument);
            _pipe->Feed(_handle->Value() + 5);
            return err_feature_ok();
        };

       private:
        std::shared_ptr<DirectInputPipe<float>> _pipe;
        std::shared_ptr<Handle<float>> _handle;
    };

   public:
    ErrorIfThreeModel() {
        _inputs.emplace("A", TypeDescriptor::Create<float>());
        _outputs.emplace("X", TypeDescriptor::Create<float>());
    }

    std::unordered_map<std::string, TypeDescriptor> const &Inputs() const override { return _inputs; }
    std::unordered_map<std::string, TypeDescriptor> const &Outputs() const override { return _outputs; }
    std::vector<std::string> GetRequirements(std::string const &outputName) const override {
        std::vector<std::string> requirements = {"A"};
        return requirements;
    }

    rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<inference::IHandle>> const &inputToHandle,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const &outputToPipe,
        std::function<void()> outOfBandNotifier) const override {
        outOfBandNotifier(); // No out of band information, so call and ignore henceforth.
        auto iterHandle = inputToHandle.find("A");
        std::shared_ptr<IHandle> handle = iterHandle == inputToHandle.end() ? nullptr : iterHandle->second;

        auto iterPipe = outputToPipe.find("X");
        std::shared_ptr<InputPipe> pipe = iterPipe == outputToPipe.end() ? nullptr : iterPipe->second;

        auto updater = std::make_shared<ValueUpdaterImpl>(handle, pipe);
        return std::static_pointer_cast<ValueUpdater>(updater);
    }

   private:
    std::unordered_map<std::string, TypeDescriptor> _inputs;
    std::unordered_map<std::string, TypeDescriptor> _outputs;
};

}  // namespace inference_test
