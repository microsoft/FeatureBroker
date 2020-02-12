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
#include <sstream>
#include <system_error>
#include <unordered_map>

using namespace ::inference;

namespace inference_test {
/// Given two inputs "int A" and "float B", have two outputs "int X = A + 5", "float Y = A + B", "std::string Z =
/// A".
class ThreeOutputModel : public Model {
   private:
    class InferenceEngineImpl : public inference::ValueUpdater {
       public:
        InferenceEngineImpl(std::shared_ptr<IHandle> const &handleA, std::shared_ptr<IHandle> const &handleB,
                            std::shared_ptr<InputPipe> const &pipeX, std::shared_ptr<InputPipe> const &pipeY,
                            std::shared_ptr<InputPipe> const &pipeZ)
            : _pipeX(std::static_pointer_cast<DirectInputPipe<int>>(pipeX)),
              _pipeY(std::static_pointer_cast<DirectInputPipe<float>>(pipeY)),
              _pipeZ(std::static_pointer_cast<DirectInputPipe<std::string>>(pipeZ)),
              _handleA(std::static_pointer_cast<Handle<int>>(handleA)),
              _handleB(std::static_pointer_cast<Handle<float>>(handleB)) {}

        std::error_code UpdateOutput() override {
            if (_pipeX != nullptr && _handleA->Changed()) _pipeX->Feed(_handleA->Value() + 5);
            if (_pipeY != nullptr && (_handleA->Changed() || _handleB->Changed()))
                _pipeY->Feed(static_cast<float>(_handleA->Value()) + _handleB->Value());
            if (_pipeZ != nullptr && _handleA->Changed()) {
                std::ostringstream str;
                str << _handleA->Value();
                _pipeZ->Feed(str.str());
            }
            return err_feature_ok();
        };

       private:
        std::shared_ptr<DirectInputPipe<int>> _pipeX;
        std::shared_ptr<DirectInputPipe<float>> _pipeY;
        std::shared_ptr<DirectInputPipe<std::string>> _pipeZ;
        std::shared_ptr<Handle<int>> _handleA;
        std::shared_ptr<Handle<float>> _handleB;
    };

   public:
    ThreeOutputModel() {
        _inputs.emplace("A", TypeDescriptor::Create<int>());
        _inputs.emplace("B", TypeDescriptor::Create<float>());

        _outputs.emplace("X", TypeDescriptor::Create<int>());
        _outputs.emplace("Y", TypeDescriptor::Create<float>());
        _outputs.emplace("Z", TypeDescriptor::Create<std::string>());
    }

    std::unordered_map<std::string, TypeDescriptor> const &Inputs() const override { return _inputs; }
    std::unordered_map<std::string, TypeDescriptor> const &Outputs() const override { return _outputs; }
    std::vector<std::string> GetRequirements(std::string const &outputName) const override {
        std::vector<std::string> requirements;
        if (outputName == "X" || outputName == "Z")
            requirements = {"A"};
        else if (outputName == "Y")
            requirements = {"A", "B"};
        return requirements;
    }

    rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<inference::IHandle>> const &inputToHandle,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const &outputToPipe,
        std::function<void()> outOfBandNotifier) const override {
        outOfBandNotifier(); // No out of band information, so call and ignore henceforth.
        auto iterHandle = inputToHandle.find("A");
        std::shared_ptr<IHandle> handleA = iterHandle == inputToHandle.end() ? nullptr : iterHandle->second;
        iterHandle = inputToHandle.find("B");
        std::shared_ptr<IHandle> handleB = iterHandle == inputToHandle.end() ? nullptr : iterHandle->second;

        auto iterPipe = outputToPipe.find("X");
        std::shared_ptr<InputPipe> pipeX = iterPipe == outputToPipe.end() ? nullptr : iterPipe->second;
        iterPipe = outputToPipe.find("Y");
        std::shared_ptr<InputPipe> pipeY = iterPipe == outputToPipe.end() ? nullptr : iterPipe->second;
        iterPipe = outputToPipe.find("Z");
        std::shared_ptr<InputPipe> pipeZ = iterPipe == outputToPipe.end() ? nullptr : iterPipe->second;

        auto updater = std::make_shared<InferenceEngineImpl>(handleA, handleB, pipeX, pipeY, pipeZ);
        return std::static_pointer_cast<ValueUpdater>(updater);
    }

   private:
    std::unordered_map<std::string, TypeDescriptor> _inputs;
    std::unordered_map<std::string, TypeDescriptor> _outputs;
};

}  // namespace inference_test
