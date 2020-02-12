// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/input_pipe.hpp>
#include <inference/value_updater.hpp>
#include <list>
#include <memory>
#include <mutex>
#include <system_error>
#include <utility>
#include <vector>

namespace inference {
class TypeDescriptor;
class IHandle;
template <typename T>
class Handle;
class OutputWaiter;

template <typename T>
class DirectInputPipe : public InputPipe {
   public:
    virtual ~DirectInputPipe() = default;

    TypeDescriptor Type() const override final;

    virtual void Feed(T) = 0;

   private:
    DirectInputPipe() = default;

    friend class FeatureBroker;             // For the DirectInputPipe<T>::Async derived class.
    friend class SynchronousFeatureBroker;  // For the DirectInputPipe<T>::SyncSingleConsumer derived class.
    friend class FeatureBrokerBase;         // For CreateHandleAndUpdater.
    friend class TypeDescriptor;            // For the DirectInputPipe<T>::SyncSingleConsumer instantiation.

    // These are nested private subclasses of DirectInputPipe<T>, declared here and defined below.
    class Async;
    class SyncSingleConsumer;

    bool _setOnce = false;
};

template <typename T>
class DirectInputPipe<T>::Async final : public DirectInputPipe<T> {
   public:
    Async();
    virtual ~Async();

    void Feed(T value) override;

   private:
    class Updater : public ValueUpdater {
       public:
        Updater(std::shared_ptr<DirectInputPipe<T>::Async> parent, std::shared_ptr<Handle<T>> handle,
                std::shared_ptr<OutputWaiter> waiter);

        virtual ~Updater();

        bool Changed() override;
        void MarkChanged(bool subsequentCall);

        std::error_code UpdateOutput() override;

       private:
        std::shared_ptr<DirectInputPipe<T>::Async> _parent;
        std::shared_ptr<Handle<T>> _handle;
        bool _changed;
        std::shared_ptr<OutputWaiter> _waiter;
    };

    std::pair<std::shared_ptr<IHandle>, std::shared_ptr<ValueUpdater>> CreateHandleAndUpdater(
        std::shared_ptr<OutputWaiter> waiter) override;

    T _value;
    std::mutex _changeMutex;
    std::list<std::weak_ptr<Updater>> _outputs;
};

template <typename T>
class DirectInputPipe<T>::SyncSingleConsumer final : public DirectInputPipe<T> {
   public:
    SyncSingleConsumer();
    virtual ~SyncSingleConsumer();
    void Feed(T value) override;

   private:
    std::pair<std::shared_ptr<IHandle>, std::shared_ptr<ValueUpdater>> CreateHandleAndUpdater(
        std::shared_ptr<InputPipe::OutputWaiter> waiter) override;

    class Updater : public ValueUpdater {
       public:
        explicit Updater(std::shared_ptr<Handle<T>> handle);
        virtual ~Updater();

        bool Changed() override;
        std::error_code UpdateOutput() override;

       private:
        std::shared_ptr<Handle<T>> _handle;
    };

    std::shared_ptr<Handle<T>> _handle;
    std::shared_ptr<Updater> _updater;
};

}  // namespace inference

#include <inference/feature_error.hpp>
#include <inference/handle.hpp>
#include <inference/type_descriptor.hpp>

namespace inference {
template <typename T>
DirectInputPipe<T>::SyncSingleConsumer::SyncSingleConsumer()
    : _handle(std::make_shared<Handle<T>>()), _updater(std::make_shared<Updater>(_handle)) {}

template <typename T>
DirectInputPipe<T>::SyncSingleConsumer::~SyncSingleConsumer() = default;

template <typename T>
void DirectInputPipe<T>::SyncSingleConsumer::Feed(T value) {
    _handle->MutableValue() = value;
    _handle->Changed(true);
    _setOnce = true;
}

template <typename T>
std::pair<std::shared_ptr<IHandle>, std::shared_ptr<ValueUpdater>>
DirectInputPipe<T>::SyncSingleConsumer::CreateHandleAndUpdater(std::shared_ptr<InputPipe::OutputWaiter> waiter) {
    if (waiter) {
        // Often this is the nullptr. When it is *not* it is because this was created from the feature-broker
        // and we are wondering whether we should "wait" or not on this input yet. We just ping the waiter
        // once and ignore it from then on.
        waiter->Ping(false);
    }
    _handle->Changed(_setOnce);
    return std::pair<std::shared_ptr<IHandle>, std::shared_ptr<ValueUpdater>>(_handle, _updater);
}

template <typename T>
DirectInputPipe<T>::SyncSingleConsumer::Updater::Updater(std::shared_ptr<Handle<T>> handle) : _handle(handle) {}

template <typename T>
DirectInputPipe<T>::SyncSingleConsumer::Updater::~Updater() = default;

template <typename T>
bool DirectInputPipe<T>::SyncSingleConsumer::Updater::Changed() {
    return _handle->Changed();
}

template <typename T>
std::error_code DirectInputPipe<T>::SyncSingleConsumer::Updater::UpdateOutput() {
    return err_feature_ok();
}

template <typename T>
DirectInputPipe<T>::Async::Async() = default;

template <typename T>
DirectInputPipe<T>::Async::~Async() = default;

template <typename T>
void DirectInputPipe<T>::Async::Feed(T value) {
    std::unique_lock<std::mutex> lock(_changeMutex);
    _value = value;
    bool subsequentSet = _setOnce;
    _setOnce = true;

    for (auto iter = _outputs.begin(); iter != _outputs.end();) {
        if (auto out = iter->lock()) {
            out->MarkChanged(subsequentSet);
            iter++;
        } else {
            iter = _outputs.erase(iter);
        }
    }
}

template <typename T>
DirectInputPipe<T>::Async::Updater::Updater(std::shared_ptr<DirectInputPipe<T>::Async> parent,
                                            std::shared_ptr<Handle<T>> handle,
                                            std::shared_ptr<InputPipe::OutputWaiter> waiter)
    : _parent(std::move(parent)), _handle(handle), _changed(_parent->_setOnce), _waiter(std::move(waiter)) {
    if (_changed) _waiter->Ping(false);
}

template <typename T>
DirectInputPipe<T>::Async::Updater::~Updater() = default;

template <typename T>
bool DirectInputPipe<T>::Async::Updater::Changed() {
    return _changed;
}

template <typename T>
void DirectInputPipe<T>::Async::Updater::MarkChanged(bool subsequentCall) {
    _changed = true;
    _waiter->Ping(subsequentCall);
}

template <typename T>
std::error_code DirectInputPipe<T>::Async::Updater::UpdateOutput() {
    if (!_changed) return err_feature_ok();

    std::lock_guard<std::mutex> lock(_parent->_changeMutex);
    _handle->MutableValue() = _parent->_value;
    _handle->Changed(true);
    _changed = false;
    return err_feature_ok();
}

template <typename T>
std::pair<std::shared_ptr<IHandle>, std::shared_ptr<ValueUpdater>> DirectInputPipe<T>::Async::CreateHandleAndUpdater(
    std::shared_ptr<InputPipe::OutputWaiter> waiter) {
    auto handle = std::make_shared<Handle<T>>();
    auto thisPtr = std::static_pointer_cast<DirectInputPipe<T>::Async>(shared_from_this());
    auto updater = std::make_shared<Updater>(thisPtr, handle, std::move(waiter));
    std::lock_guard<std::mutex> lock(_changeMutex);
    _outputs.push_back(updater);
    return std::pair<std::shared_ptr<IHandle>, std::shared_ptr<ValueUpdater>>(handle, updater);
}

template <typename T>
TypeDescriptor DirectInputPipe<T>::Type() const {
    // Because pipes are not publicly constructable, this should succeed.
    return TypeDescriptor::_CreateUnsafe<T>();
}
}  // namespace inference
