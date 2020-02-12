// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <condition_variable>
#include <inference/feature_broker.hpp>
#include <mutex>

namespace inference {

static bool HandleChanged(std::shared_ptr<const IHandle> const& handle) { return handle->Changed(); }
static bool UpdaterChanged(std::shared_ptr<ValueUpdater> const& updater) { return updater->Changed(); }

std::error_code FeatureBrokerBase::CheckInputOk(std::string const& name,
                                                rt::expected<TypeDescriptor> const& typeDescriptorExpected) const
    noexcept {
    if (!typeDescriptorExpected) return typeDescriptorExpected.error();
    auto pipeType = typeDescriptorExpected.value();

    auto model = GetModelOrNull(false);
    if (model != nullptr) {
        // If the model is bound, check to see if this input is a required
        // input. If it is a required input confirm the type is correct -- if
        // not return type mismatch. Note this only happens when the model is
        // bound, otherwise the input would be accepted as-is.
        auto inputsIter = model->Inputs().find(name);
        if (inputsIter != model->Inputs().end() && inputsIter->second != pipeType)
            return make_feature_error(feature_errc::type_mismatch);
    }
    // Check to see if the input is already bound.
    auto typeExpected = GetBindingType(name, false);
    if (typeExpected) return make_feature_error(feature_errc::already_bound);
    if (typeExpected.error() != feature_errc::not_bound) return typeExpected.error();
    return std::error_code();
}

std::error_code FeatureBrokerBase::CheckModelOutput(std::string const& name,
                                                    rt::expected<TypeDescriptor> const& typeDescriptorExpected) const
    noexcept {
    auto model = GetModelOrNull();
    if (model == nullptr) return make_feature_error(feature_errc::no_model_associated);
    auto outputIter = model->Outputs().find(name);
    if (outputIter == model->Outputs().end()) {
        return make_feature_error(feature_errc::name_not_found);
    }
    if (!typeDescriptorExpected) return typeDescriptorExpected.error();
    if (outputIter->second != typeDescriptorExpected.value()) return make_feature_error(feature_errc::type_mismatch);
    return err_feature_ok();
}

// Pass by const ref because these were already passed by value in the derived classes.
FeatureBrokerBase::FeatureBrokerBase(std::shared_ptr<const Model> model) : _model(std::move(model)) {}

std::shared_ptr<const Model> FeatureBrokerBase::GetModelOrNull(bool lock) const { return _model; }

const rt::expected<TypeDescriptor> FeatureBrokerBase::GetBindingType(std::string const& name, bool lock) const {
    std::shared_lock<std::shared_mutex> localLock;
    if (lock) localLock = std::shared_lock<std::shared_mutex>(_inputMutex);
    if (auto inputPipe = this->GetBindingOrNull(name, false)) return inputPipe->Type();
    if (auto provider = this->GetProviderOrNull(name, false)) {
        auto found = provider->Outputs().find(name);
        // This could only happen if the implementation of the feature provider was mutating state past being bound.
        // This would be a serious bug on the part of the client code.
        if (found == provider->Outputs().end()) return make_feature_unexpected(feature_errc::feature_provider_inconsistent);
        return found->second;
    }
    // Despite the name, not necessarily an error condition, depending on the context.
    return make_feature_unexpected(feature_errc::not_bound);
}

std::shared_ptr<InputPipe> FeatureBrokerBase::GetBindingOrNull(std::string const& name, bool lock) const {
    std::shared_lock<std::shared_mutex> localLock;
    if (lock) localLock = std::shared_lock<std::shared_mutex>(_inputMutex);
    auto found = _boundInputs.find(name);
    if (found != _boundInputs.end()) return found->second;
    return {};
}

std::shared_ptr<FeatureProvider> FeatureBrokerBase::GetProviderOrNull(std::string const& name, bool lock) const {
    std::shared_lock<std::shared_mutex> localLock;
    if (lock) localLock = std::shared_lock<std::shared_mutex>(_inputMutex);
    auto found = _boundInputsFromProviders.find(name);

    if (found != _boundInputsFromProviders.end()) return found->second;
    return {};
}

rt::expected<void> FeatureBrokerBase::BindCore(std::string const& name, std::shared_ptr<InputPipe> pipe) {
    // Under these circumstances, merely not being bound is not an error. But, anything else is.
    std::unique_lock<std::shared_mutex> lock(_inputMutex);
    if (auto error = CheckInputOk(name, pipe->Type())) return tl::unexpected(error);
    _boundInputs.emplace(name, pipe);
    return {};
}

rt::expected<void> FeatureBrokerBase::BindInputs(std::shared_ptr<FeatureProvider> provider) {
    std::unique_lock<std::shared_mutex> lock(_inputMutex);
    for (auto& pair : provider->Outputs()) {
        if (auto error = CheckInputOk(pair.first, pair.second)) return tl::unexpected(error);
    }
    // Now that we've checked that it's OK, add it to the bindings.
    for (auto& pair : provider->Outputs()) {
        _boundInputsFromProviders.emplace(pair.first, provider);
    }
    return {};
}

FeatureBrokerBase::BrokerOutputPipeGeneral::BrokerOutputPipeGeneral() = default;

class OutputWaiterSinglePing final {
   public:
    OutputWaiterSinglePing(std::shared_ptr<InputPipe::OutputWaiter> waiter) : _waiter(std::move(waiter)) {}
    void Ping() {
        _waiter->Ping(_subsequentCall);
        _subsequentCall = true;
    }

   private:
    bool _subsequentCall{false};
    std::shared_ptr<InputPipe::OutputWaiter> _waiter;
};

std::error_code FeatureBrokerBase::BrokerOutputPipeGeneral::Bind(FeatureBrokerBase& featureBroker,
                                                                 std::vector<std::string> const& outputNames) {
    auto model = featureBroker.GetModelOrNull();
    std::map<std::string, std::shared_ptr<InputPipe>> inputNameToPipe;
    std::unordered_map<FeatureProvider*, std::unordered_set<std::string>> providerToRequested;

    for (auto& outputName : outputNames) {
        auto inputs = model->GetRequirements(outputName);
        for (auto& inputName : inputs) {
            auto inputIter = model->Inputs().find(inputName);
            if (inputIter == model->Inputs().end()) return make_feature_error(feature_errc::name_not_found);
            // If we already found the bound input, so no need to repeat it.
            if (inputNameToPipe.find(inputName) != inputNameToPipe.end()) continue;

            auto provider = featureBroker.GetProviderOrNull(inputName);
            if (provider != nullptr) {
                // This input was bound by a provider. First check the type of the provider.
                auto providerPair = provider->Outputs().find(inputName);
                if (providerPair == provider->Outputs().end())
                    return make_feature_error(feature_errc::feature_provider_inconsistent);
                if (providerPair->second != inputIter->second) return make_feature_error(feature_errc::type_mismatch);
                providerToRequested[provider.get()].insert(inputName);
            } else if (auto inputPipe = featureBroker.GetBindingOrNull(inputName)) {
                // This input was bound by a pipe.
                if (inputPipe->Type() != inputIter->second) return make_feature_error(feature_errc::type_mismatch);
                inputNameToPipe.emplace(inputName, inputPipe);
            } else {
                return make_feature_error(feature_errc::not_bound);
            }
        }
    }

    // The number of waiters will be the number of input pipes, plus the number of providers,
    // plus one more for the model itself.
    auto outputWaiter =
        std::make_shared<InputPipe::OutputWaiter>(inputNameToPipe.size() + providerToRequested.size() + 1);

    // Now that we've verified that the bindings are complete and compatible,
    // set up the structures necessary to do the inference. Start with the pipes...

    for (auto& inputPair : inputNameToPipe) {
        std::string inputName = inputPair.first;
        std::shared_ptr<InputPipe> inputPipe = inputPair.second;

        auto pair = inputPipe->CreateHandleAndUpdater(outputWaiter);
        auto handle = pair.first;
        auto updater = pair.second;

        _inputToHandle.emplace(inputName, handle);
        _handlesForInputs.push_back(handle);
        _updatersForInputs.push_back(updater);
    }

    // Continue with the providers.
    for (auto& inputPair : providerToRequested) {
        auto providerPtr = inputPair.first;
        auto& requestedInputs = inputPair.second;

        // Form the map.
        std::map<std::string, std::shared_ptr<InputPipe>> nameToPipe;
        for (auto& inputName : requestedInputs) {
            auto inputPipe = providerPtr->Outputs().at(inputName).CreateDirectInputPipeSyncSingleConsumer();
            nameToPipe.emplace(inputName, inputPipe);
            // Passing in the nullptr is fine in this case since we know it is a synchronous pipe.
            auto pair = inputPipe->CreateHandleAndUpdater(nullptr);
            _inputToHandle.emplace(inputName, pair.first);
            _handlesForInputs.push_back(pair.first);
        }

        // Using this map, feed it to the provider and get its updater.
        // We must use C++11, but with C++14 we could avoid the creation of this shared variable captured by value
        // through generalized lambda capture.
        auto singlePing = std::make_shared<OutputWaiterSinglePing>(outputWaiter);
        auto updaterExpected = providerPtr->CreateValueUpdater(nameToPipe, [singlePing]() { singlePing->Ping(); });
        if (!updaterExpected) return updaterExpected.error();
        _updatersForInputs.push_back(updaterExpected.value());
    }

    std::map<std::string, std::shared_ptr<InputPipe>> outputToInputPipe;

    for (auto outputName : outputNames) {
        auto outputIter = model->Outputs().find(outputName);
        // The following should not happen since we already iterated over this.
        if (outputIter == model->Outputs().end()) return make_feature_error(feature_errc::name_not_found);

        auto inputPipe = outputIter->second.CreateDirectInputPipeSyncSingleConsumer();
        // Similar to above, since synchronous the waiter is not relevant so can pass in nullptr.
        auto inputHandleAndUpdater = inputPipe->CreateHandleAndUpdater(nullptr);
        _handlesForOutputs.push_back(inputHandleAndUpdater.first);
        _updatersForOutputs.push_back(inputHandleAndUpdater.second);
        outputToInputPipe.emplace(outputName, inputPipe);
    }

    auto singlePing = std::make_shared<OutputWaiterSinglePing>(outputWaiter);
    auto updaterExpected =
        model->CreateValueUpdater(_inputToHandle, outputToInputPipe, [singlePing]() { singlePing->Ping(); });
    if (!updaterExpected) return updaterExpected.error();
    _spEngineForOutput = updaterExpected.value();
    _waiter = std::move(outputWaiter);
    return {};
}

bool FeatureBrokerBase::BrokerOutputPipeGeneral::ChangedImpl() const {
    if (!_waiter->Cleared()) return false;
    if (_firstOutputFetched) {
        return std::any_of(_updatersForInputs.begin(), _updatersForInputs.end(), UpdaterChanged);
    }
    return std::all_of(_updatersForInputs.begin(), _updatersForInputs.end(), UpdaterChanged);
}

bool FeatureBrokerBase::BrokerOutputPipeGeneral::UpdateIfChangedPrePeek() {
    // In principle all of these should succeed since the implementations are internal.
    if (!ChangedImpl()) return false;
    for (auto& updater : _updatersForInputs) updater->UpdateOutput();
    if (_firstOutputFetched) {
        if (!std::any_of(_handlesForInputs.begin(), _handlesForInputs.end(), HandleChanged)) return false;
    } else {
        if (!std::all_of(_handlesForInputs.begin(), _handlesForInputs.end(), HandleChanged)) return false;
    }
    // Set the output pipes to unchanged so that we can detect whether the model's updater actually updated those
    // pipes.
    for (auto& handle : _handlesForOutputs) handle->Changed(false);
    return true;
}

rt::expected<bool> FeatureBrokerBase::BrokerOutputPipeGeneral::UpdateIfChangedInference() {
    // Now query the model.
    if (!_spEngineForOutput->Changed()) return false;
    // Note that this update may potentially fail.
    auto errc = _spEngineForOutput->UpdateOutput();
    if (errc) return tl::make_unexpected(errc);

    // As with the input updaters, these should all succeed.
    for (auto& outputUpdaters : _updatersForOutputs) outputUpdaters->UpdateOutput();
    _firstOutputFetched = true;

    return std::any_of(_handlesForOutputs.begin(), _handlesForOutputs.end(), HandleChanged);
}

void FeatureBrokerBase::BrokerOutputPipeGeneral::UpdateIfChangedPostPoke() {
    // We do have an updated output. Set all the inputs to consumed.
    for (auto& handle : _handlesForInputs) handle->Changed(false);
}

rt::expected<void> FeatureBrokerBase::BrokerOutputPipeGeneral::WaitUntilChangedImpl() { return tl::make_unexpected(_waiter->Wait()); }

}  // namespace inference
