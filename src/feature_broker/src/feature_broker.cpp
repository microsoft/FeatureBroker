#include <inference/feature_broker.hpp>

namespace {
// Because SetParent can introduce non-local looping structures, in order to make FeatureBroker::SetParent concurrency
// safe, we actually want a more-or-less global exclusive lock for SetParent to be held. Attempting to rely on the local
// locks to each feature broker too easily results in deadlocks. If we did not have to check for loops on parentage.
std::mutex setParentMutex;

}  // namespace

namespace inference {

FeatureBroker::FeatureBroker(std::shared_ptr<const Model> model) : FeatureBrokerBase(model) {}

FeatureBroker::~FeatureBroker() {}

FeatureBroker::FeatureBroker(std::shared_ptr<const FeatureBroker> parent, std::shared_ptr<const Model> model)
    : FeatureBrokerBase(model), _parent(std::move(parent)) {}

rt::expected<std::shared_ptr<FeatureBroker>> FeatureBroker::Fork(std::shared_ptr<const Model> model) const {
    if (model) {
        std::shared_lock<std::shared_mutex> localLock(_inputMutex);
        for (auto pair : model->Inputs()) {
            auto existingType = GetBindingType(pair.first, false);
            if (!existingType) continue;
            if (existingType.value() != pair.second) return make_feature_unexpected(feature_errc::type_mismatch);
        }
    }
    return std::shared_ptr<FeatureBroker>(new FeatureBroker(shared_from_this(), std::move(model)));
}

rt::expected<void> FeatureBroker::SetParent(std::shared_ptr<const FeatureBroker> newParent) {
    // In this trivial case, we consider this a no-op.
    if (newParent.get() == this) return {};

    std::unique_lock<std::mutex> globalLock(::setParentMutex);
    std::unique_lock<std::shared_mutex> lock(_inputMutex);

    // Ensure no cycles, that is, that this object is never an ancestor. The only way that a cycle can be introduced is
    // if the new parent has as an ancestor this object.
    for (auto newAncestor = newParent; newAncestor; newAncestor = newAncestor->_parent) {
        if (newAncestor.get() == this) return make_feature_unexpected(feature_errc::circular_structure);
    }
    // We need to ensure there are no inconsistencies in its or our already bound inputs with our model; that is, if we
    // have one.
    if (newParent) {
        // Make sure there are no conflicting bindings. Note that we should not use the convenience functions directly
        // since they will traverse the existing hierarchy, and that may change.
        for (const auto& namePipe : _boundInputs) {
            if (newParent->GetBindingType(namePipe.first)) return make_feature_unexpected(feature_errc::already_bound);
        }
        for (const auto& nameProvider : _boundInputsFromProviders) {
            if (newParent->GetBindingType(nameProvider.first)) return make_feature_unexpected(feature_errc::already_bound);
        }
        // Now that we've verified there are no bind conflicts, check the model inputs, if we have a local model. Note
        // that for the same reasons as elsewhere we don't use the GetModelOrNull accessors since that will traverse the
        // existing hierarchy, which we are (in principle) about to replace.
        if (_model) {
            for (const auto& nameType : _model->Inputs()) {
                auto typeExpected = newParent->GetBindingType(nameType.first);
                if (typeExpected && typeExpected.value() != nameType.second)
                    return make_feature_unexpected(feature_errc::type_mismatch);
            }
        } else {
            auto parentModel = newParent->GetModelOrNull();
            if (parentModel) {
                // If we don't have a local model but the parent does, we have to ensure that our already bound inputs
                // don't have type mismatches on its new parent's model. Note the usage of the base class methods to
                // ensure we don't traverse the existing parents.
                for (const auto& nameType : parentModel->Inputs()) {
                    auto inputPipe = FeatureBrokerBase::GetBindingOrNull(nameType.first, false);
                    if (inputPipe && (inputPipe->Type() != nameType.second))
                        return make_feature_unexpected(feature_errc::type_mismatch);
                    auto provider = FeatureBrokerBase::GetProviderOrNull(nameType.first, false);
                    if (provider) {
                        auto found = provider->Outputs().find(nameType.first);
                        if (found != provider->Outputs().end() && found->second != nameType.second)
                            return make_feature_unexpected(feature_errc::type_mismatch);
                    }
                }
            }
        }
    }
    // All is well. Set the parent.
    _parent = std::move(newParent);
    return {};
}

std::shared_ptr<const Model> FeatureBroker::GetModelOrNull(bool lock) const {
    std::shared_lock<std::shared_mutex> localLock;
    if (lock) localLock = std::shared_lock<std::shared_mutex>(_inputMutex);
    if (_model || !_parent) return _model;
    return _parent->GetModelOrNull();
}

std::shared_ptr<InputPipe> FeatureBroker::GetBindingOrNull(std::string const& name, bool lock) const {
    std::shared_lock<std::shared_mutex> localLock;
    if (lock) localLock = std::shared_lock<std::shared_mutex>(_inputMutex);
    auto inputPipe = FeatureBrokerBase::GetBindingOrNull(name, false);
    if (inputPipe || !_parent) return inputPipe;
    return _parent->GetBindingOrNull(name);
}

std::shared_ptr<FeatureProvider> FeatureBroker::GetProviderOrNull(std::string const& name, bool lock) const {
    std::shared_lock<std::shared_mutex> localLock;
    if (lock) localLock = std::shared_lock<std::shared_mutex>(_inputMutex);
    auto provider = FeatureBrokerBase::GetProviderOrNull(name, false);
    if (provider || !_parent) return provider;
    return _parent->GetProviderOrNull(name);
}

}  // namespace inference
