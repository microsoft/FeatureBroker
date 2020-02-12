// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/input_pipe.hpp>
#include <inference/model.hpp>
#include <inference/type_descriptor.hpp>
#include <inference/value_updater.hpp>
#include <map>
#include <memory>
#include <system_error>
#include <unordered_map>

using namespace ::inference;

namespace inference_test {
/**
 * A meta-model that wraps a model, and allows the dferment of the call of the
 * notification function until a release method is called.
 */
class ReleaseModel : public Model {
   private:
    class ValueUpdaterImpl : public ValueUpdater {
       public:
        ValueUpdaterImpl(std::function<void()> notifier)
            : _internalNotifierCalled(false), _myNofierCalled(false), _notifier(notifier) {}

        void Set(std::shared_ptr<ValueUpdater> updater) { _updater = std::move(updater); }
        std::error_code UpdateOutput() override { return _updater->UpdateOutput(); };

        void InternalNotifier() { NotifierCore(_internalNotifierCalled); }
        void Release() { NotifierCore(_myNofierCalled); }

       private:
        std::mutex _mutex;
        bool _internalNotifierCalled;
        bool _myNofierCalled;
        const std::function<void()> _notifier;
        std::shared_ptr<ValueUpdater> _updater;

        void NotifierCore(bool &toSet) {
            std::unique_lock<std::mutex> lock(_mutex);
            if (toSet == true) return;
            toSet = true;
            if (_myNofierCalled && _internalNotifierCalled) _notifier();
        }
    };

   public:
    ReleaseModel(std::shared_ptr<const Model> model) : _model(std::move(model)), _releaseCalledOnce(false) {}

    std::unordered_map<std::string, TypeDescriptor> const &Inputs() const override { return _model->Inputs(); }
    std::unordered_map<std::string, TypeDescriptor> const &Outputs() const override { return _model->Outputs(); }
    std::vector<std::string> GetRequirements(std::string const &outputName) const override {
        return _model->GetRequirements(outputName);
    }

    rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<inference::IHandle>> const &inputToHandle,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const &outputToPipe,
        std::function<void()> outOfBandNotifier) const override {
        auto updater = std::make_shared<ValueUpdaterImpl>(outOfBandNotifier);
        auto internalUpdaterExpected =
            _model->CreateValueUpdater(inputToHandle, outputToPipe, [&updater]() { updater->InternalNotifier(); });
        if (!internalUpdaterExpected) return tl::make_unexpected(internalUpdaterExpected.error());
        updater->Set(internalUpdaterExpected.value());
        std::unique_lock<std::mutex> lock(_mutex);
        if (_releaseCalledOnce)
            updater->Release();
        else
            _updaters.push_back(updater);
        return std::static_pointer_cast<ValueUpdater>(updater);
    }

    void Release() {
        std::unique_lock<std::mutex> lock(_mutex);
        for (auto iter = _updaters.begin(); iter != _updaters.end(); ++iter) {
            if (auto updater = iter->lock()) {
                updater->Release();
            }
        }
        _updaters.clear();
        _releaseCalledOnce = true;
    }

   private:
    const std::shared_ptr<const Model> _model;
    mutable std::mutex _mutex;
    bool _releaseCalledOnce;
    mutable std::vector<std::weak_ptr<ValueUpdaterImpl>> _updaters;
};

}  // namespace inference_test
