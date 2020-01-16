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
class AddValueUpdater : public inference::ValueUpdater {
   public:
    AddValueUpdater() {}

    AddValueUpdater(std::shared_ptr<IHandle> const& aHandle, std::shared_ptr<IHandle> const& bHandle,
                    std::shared_ptr<InputPipe> const& pipe)
        : _pipe(std::static_pointer_cast<DirectInputPipe<float>>(pipe)),
          _aHandle(std::static_pointer_cast<Handle<float>>(aHandle)),
          _bHandle(std::static_pointer_cast<Handle<float>>(bHandle)) {}

    std::error_code UpdateOutput() override {
        auto value = _aHandle->Value() + _bHandle->Value();
        _pipe->Feed(value);
        return err_feature_ok();
    }

   private:
    std::shared_ptr<DirectInputPipe<float>> _pipe;
    std::shared_ptr<Handle<float>> _aHandle;
    std::shared_ptr<Handle<float>> _bHandle;
};

/// <summary>
/// Given 'A' and 'B' as inputs, adds the two numbers and publishes as the output 'X'
/// </summary>
class AddModel : public Model {
   public:
    AddModel() {
        _inputs.emplace("A", TypeDescriptor::Create<float>());
        _inputs.emplace("B", TypeDescriptor::Create<float>());
        _outputs.emplace("X", TypeDescriptor::Create<float>());
    }

    ~AddModel() {}

    std::unordered_map<std::string, TypeDescriptor> const& Inputs() const override { return _inputs; }

    std::unordered_map<std::string, TypeDescriptor> const& Outputs() const override { return _outputs; }

    std::vector<std::string> GetRequirements(std::string const& outputName) const override {
        std::vector<std::string> requirements = {"A", "B"};
        return requirements;
    }

    rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe,
        std::function<void()> outOfBandNotifier) const override {
        outOfBandNotifier(); // No out of band information, so call and ignore henceforth.
        auto iterPipe = outputToPipe.find("X");
        if (iterPipe == outputToPipe.end()) {
            return make_feature_unexpected(inference::feature_errc::name_not_found);
        }

        auto iterHandle = inputToHandle.find("A");
        if (iterHandle == inputToHandle.end()) {
            return make_feature_unexpected(inference::feature_errc::name_not_found);
        }
        auto aHandle = iterHandle->second;

        iterHandle = inputToHandle.find("B");
        if (iterHandle == inputToHandle.end()) {
            return make_feature_unexpected(inference::feature_errc::name_not_found);
        }

        auto bHandle = iterHandle->second;

        std::shared_ptr<InputPipe> pipe = iterPipe->second;
        auto updater = std::make_shared<AddValueUpdater>(aHandle, bHandle, pipe);
        return std::static_pointer_cast<ValueUpdater>(updater);
    }

   private:
    std::unordered_map<std::string, TypeDescriptor> _inputs;
    std::unordered_map<std::string, TypeDescriptor> _outputs;
};

}  // namespace inference_test
