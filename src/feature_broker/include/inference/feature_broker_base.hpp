// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <inference/direct_input_pipe.hpp>
#include <inference/feature_error.hpp>
#include <inference/feature_provider.hpp>
#include <inference/handle.hpp>
#include <inference/input_pipe.hpp>
#include <inference/model.hpp>
#include <inference/output_pipe.hpp>
#include <inference/output_pipe_with_input.hpp>
#include <inference/type_descriptor.hpp>
#include <inference/value_updater.hpp>
#include <initializer_list>
#include <map>
#include <memory>
#include <rt/rt_expected.hpp>
#include <shared_mutex>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "feature_broker_export.h"

namespace inference {
class FeatureBrokerBase {
   private:
    using InputsType = std::map<std::string, std::shared_ptr<IHandle>>;

    template <typename T>
    rt::expected<std::shared_ptr<OutputPipeWithInput<T, InputsType>>> _BindOutput(std::string const &name);

    template <class... T>
    rt::expected<std::shared_ptr<OutputPipeWithInput<std::tuple<T...>, InputsType>>> _BindOutputs(
        const std::initializer_list<std::string> names);

    friend class FeatureBroker;
    friend class SynchronousFeatureBroker;
    FEATURE_BROKER_EXPORT explicit FeatureBrokerBase(std::shared_ptr<const Model> model);

    template <typename TPipe, typename TPipeImplementation, typename TData>
    rt::expected<std::shared_ptr<TPipe>> BindCore(std::string const &name);

    FEATURE_BROKER_EXPORT rt::expected<void> BindCore(std::string const &name, std::shared_ptr<InputPipe> pipe);

    FEATURE_BROKER_EXPORT rt::expected<void> BindInputs(std::shared_ptr<FeatureProvider> provider);

    virtual std::shared_ptr<const Model> GetModelOrNull(bool lock = true) const;
    FEATURE_BROKER_EXPORT const rt::expected<TypeDescriptor> GetBindingType(std::string const &name,
                                                                            bool lock = true) const;
    virtual std::shared_ptr<InputPipe> GetBindingOrNull(std::string const &name, bool lock = true) const;
    virtual std::shared_ptr<FeatureProvider> GetProviderOrNull(std::string const &name, bool lock = true) const;

    class BrokerOutputPipeGeneral {
       public:
        FEATURE_BROKER_EXPORT BrokerOutputPipeGeneral();
        virtual ~BrokerOutputPipeGeneral() = default;
        FEATURE_BROKER_EXPORT std::error_code Bind(FeatureBrokerBase &featureBroker,
                                                   std::vector<std::string> const &outputNames);
        FEATURE_BROKER_EXPORT bool ChangedImpl() const;

       protected:
        std::vector<std::shared_ptr<IHandle>> _handlesForOutputs;
        InputsType _inputToHandle;

        FEATURE_BROKER_EXPORT bool UpdateIfChangedPrePeek();
        FEATURE_BROKER_EXPORT rt::expected<bool> UpdateIfChangedInference();
        FEATURE_BROKER_EXPORT void UpdateIfChangedPostPoke();
        FEATURE_BROKER_EXPORT rt::expected<void> WaitUntilChangedImpl();

       private:
        bool _firstOutputFetched{false};
        std::vector<std::shared_ptr<IHandle>> _handlesForInputs;
        std::vector<std::shared_ptr<ValueUpdater>> _updatersForInputs;

        std::shared_ptr<ValueUpdater> _spEngineForOutput;

        // We need these to update the _handlesForOutputs, since we are using the async capable DirectInputPipe for this
        // purpose. Once we switch to another input pipe that does not have async support (since even in async scenarios
        // this is not necessary for this purpose), we do not need to keep this around any longer.
        std::vector<std::shared_ptr<ValueUpdater>> _updatersForOutputs;
        std::shared_ptr<InputPipe::OutputWaiter> _waiter;
    };

    template <typename T>
    class BrokerOutputPipe : public OutputPipeWithInput<T, InputsType>, protected BrokerOutputPipeGeneral {
       public:
        BrokerOutputPipe() = default;
        virtual ~BrokerOutputPipe() = default;

        bool Changed() override;
        rt::expected<bool> UpdateIfChanged(T &value) override;
        rt::expected<void> WaitUntilChanged() override;
        const InputsType &Inputs() override { return _inputToHandle; }

       protected:
        virtual void Peek(T &value) = 0;
        virtual void Poke(T &value) const = 0;
    };

    template <typename T>
    class SingleValueOutputPipe final : public BrokerOutputPipe<T> {
       public:
        SingleValueOutputPipe() = default;
        std::error_code Bind(FeatureBrokerBase &featureBroker, std::string const &outputName);

       protected:
        void Peek(T &value) override;
        void Poke(T &value) const override;
    };

    template <class... T>
    class TupleOutputPipe final : public BrokerOutputPipe<std::tuple<T...>> {
       public:
        TupleOutputPipe() = default;
        std::error_code Bind(FeatureBrokerBase &featureBroker, std::vector<std::string> const &outputNames);

       protected:
        void Peek(std::tuple<T...> &value) override;
        void Poke(std::tuple<T...> &value) const override;

       private:
        // general specification.
        template <size_t K, class... TTuple>
        struct Unpacker {
            // Ensure that any instantiation is one of the specializations below.
            Unpacker() = delete;
        };

        // Base case, with no args left.
        template <size_t K>
        struct Unpacker<K> {
            std::error_code BindTypeCheck(FeatureBrokerBase &featureBroker,
                                          std::vector<std::string> const &outputNames) const {
                return err_feature_ok();
            }
            void Peek(std::vector<std::shared_ptr<IHandle>> const &handles, std::tuple<T...> &value) {}
            void Poke(std::vector<std::shared_ptr<IHandle>> const &handles, std::tuple<T...> &value) const {}
        };

        // Recursive case.
        template <size_t K, class THead, class... TRest>
        struct Unpacker<K, THead, TRest...> : Unpacker<K + 1, TRest...> {
            std::error_code BindTypeCheck(FeatureBrokerBase &featureBroker,
                                          std::vector<std::string> const &outputNames) const;
            void Peek(std::vector<std::shared_ptr<IHandle>> const &handles, std::tuple<T...> &value);
            void Poke(std::vector<std::shared_ptr<IHandle>> const &handles, std::tuple<T...> &value) const;
        };

        Unpacker<0, T...> _unpacker;
    };

    // Have a private deleted cctor to avoid copying.
    FeatureBrokerBase(const FeatureBrokerBase &other) = delete;

    // If there is a model, checks whether there is an input, and if so whether the type matches.
    FEATURE_BROKER_EXPORT std::error_code CheckInputOk(std::string const &name,
                                                       rt::expected<TypeDescriptor> const &typeDescriptorExpected) const
        noexcept;
    template <typename T>
    std::error_code CheckInputOk(std::string const &name) const noexcept {
        return CheckInputOk(name, TypeDescriptor::CreateExpected<T>());
    }

    // Ensures that there is a model, ensures it has an output with this name, and ensures that the types match.
    FEATURE_BROKER_EXPORT std::error_code CheckModelOutput(
        std::string const &name, rt::expected<TypeDescriptor> const &typeDescriptorExpected) const noexcept;
    template <typename T>
    std::error_code CheckModelOutput(std::string const &name) const noexcept {
        return CheckModelOutput(name, TypeDescriptor::CreateExpected<T>());
    }

    mutable std::shared_mutex _inputMutex;
    std::map<std::string, std::shared_ptr<InputPipe>> _boundInputs;
    std::map<std::string, std::shared_ptr<FeatureProvider>> _boundInputsFromProviders;
    std::shared_ptr<const Model> _model;
};

template <typename T>
rt::expected<std::shared_ptr<OutputPipeWithInput<T, FeatureBrokerBase::InputsType>>> FeatureBrokerBase::_BindOutput(
    std::string const &name) {
    std::error_code error = CheckModelOutput<T>(name);
    if (error) return tl::make_unexpected(error);
    auto brokerOutputPipe = std::make_shared<SingleValueOutputPipe<T>>();
    if ((error = brokerOutputPipe->Bind(*this, name))) return tl::make_unexpected(error);
    return std::static_pointer_cast<OutputPipeWithInput<T, FeatureBrokerBase::InputsType>>(brokerOutputPipe);
}

template <class... T>
rt::expected<std::shared_ptr<OutputPipeWithInput<std::tuple<T...>, FeatureBrokerBase::InputsType>>>
FeatureBrokerBase::_BindOutputs(const std::initializer_list<std::string> names) {
    if (names.size() != std::tuple_size<std::tuple<T...>>())
        return make_feature_unexpected(feature_errc::invalid_operation);

    auto brokerOutputPipe = std::make_shared<TupleOutputPipe<T...>>();
    std::vector<std::string> namesVec = names;
    if (auto error = brokerOutputPipe->Bind(*this, namesVec)) return tl::make_unexpected(error);
    return std::static_pointer_cast<OutputPipeWithInput<std::tuple<T...>, FeatureBrokerBase::InputsType>>(
        brokerOutputPipe);
}

template <typename TPipe, typename TPipeImplementation, typename TData>
rt::expected<std::shared_ptr<TPipe>> FeatureBrokerBase::BindCore(std::string const &name) {
    std::error_code error;
    auto requestedTypeExpected = TypeDescriptor::CreateExpected<TData>();
    if (!requestedTypeExpected) return tl::make_unexpected(requestedTypeExpected.error());
    auto input = std::make_shared<TPipeImplementation>();
    auto expected = BindCore(name, input);
    if (!expected) return tl::make_unexpected(expected.error());
    return std::static_pointer_cast<TPipe>(input);
}

template <typename T>
bool FeatureBrokerBase::BrokerOutputPipe<T>::Changed() {
    return BrokerOutputPipeGeneral::ChangedImpl();
}

template <typename T>
rt::expected<bool> FeatureBrokerBase::BrokerOutputPipe<T>::UpdateIfChanged(T &value) {
    if (!BrokerOutputPipeGeneral::UpdateIfChangedPrePeek()) return false;
    Peek(value);
    auto inferenceExpected = BrokerOutputPipeGeneral::UpdateIfChangedInference();
    if (!(inferenceExpected && inferenceExpected.value())) return inferenceExpected;
    Poke(value);
    BrokerOutputPipeGeneral::UpdateIfChangedPostPoke();
    return true;
}

template <typename T>
rt::expected<void> FeatureBrokerBase::BrokerOutputPipe<T>::WaitUntilChanged() {
    return BrokerOutputPipeGeneral::WaitUntilChangedImpl();
}

// SINGLE

template <typename T>
std::error_code FeatureBrokerBase::SingleValueOutputPipe<T>::Bind(FeatureBrokerBase &featureBroker,
                                                                  std::string const &outputName) {
    const std::vector<std::string> names = {outputName};
    return FeatureBrokerBase::BrokerOutputPipeGeneral::Bind(featureBroker, names);
}

template <typename T>
void FeatureBrokerBase::SingleValueOutputPipe<T>::Peek(T &value) {
    auto untypedHandle = FeatureBrokerBase::BrokerOutputPipeGeneral::_handlesForOutputs[0];
    auto handle = std::static_pointer_cast<Handle<T>>(untypedHandle);
    handle->MutableValue() = value;
}

template <typename T>
void FeatureBrokerBase::SingleValueOutputPipe<T>::Poke(T &value) const {
    auto untypedHandle = FeatureBrokerBase::BrokerOutputPipeGeneral::_handlesForOutputs[0];
    auto handle = std::static_pointer_cast<Handle<T>>(untypedHandle);
    value = handle->Value();
}

// TUPLE

template <class... T>
std::error_code FeatureBrokerBase::TupleOutputPipe<T...>::Bind(FeatureBrokerBase &featureBroker,
                                                               std::vector<std::string> const &outputNames) {
    // First check the types.
    auto error = _unpacker.BindTypeCheck(featureBroker, outputNames);
    if (error) return error;
    return FeatureBrokerBase::BrokerOutputPipeGeneral::Bind(featureBroker, outputNames);
}

template <class... T>
void FeatureBrokerBase::TupleOutputPipe<T...>::Peek(std::tuple<T...> &value) {
    _unpacker.Peek(FeatureBrokerBase::BrokerOutputPipe<std::tuple<T...>>::_handlesForOutputs, value);
}

template <class... T>
void FeatureBrokerBase::TupleOutputPipe<T...>::Poke(std::tuple<T...> &value) const {
    _unpacker.Poke(FeatureBrokerBase::BrokerOutputPipe<std::tuple<T...>>::_handlesForOutputs, value);
}

template <class... T>
template <size_t K, class THead, class... TRest>
std::error_code FeatureBrokerBase::TupleOutputPipe<T...>::Unpacker<K, THead, TRest...>::BindTypeCheck(
    FeatureBrokerBase &featureBroker, std::vector<std::string> const &outputNames) const {
    std::error_code error = featureBroker.CheckModelOutput<THead>(outputNames.at(K));
    if (error) return error;
    return Unpacker<K + 1, TRest...>::BindTypeCheck(featureBroker, outputNames);
}

template <class... T>
template <size_t K, class THead, class... TRest>
void FeatureBrokerBase::TupleOutputPipe<T...>::Unpacker<K, THead, TRest...>::Peek(
    std::vector<std::shared_ptr<IHandle>> const &handles, std::tuple<T...> &value) {
    auto handle = std::static_pointer_cast<Handle<THead>>(handles.at(K));
    handle->MutableValue() = std::get<K>(value);
    Unpacker<K + 1, TRest...>::Peek(handles, value);
}

template <class... T>
template <size_t K, class THead, class... TRest>
void FeatureBrokerBase::TupleOutputPipe<T...>::Unpacker<K, THead, TRest...>::Poke(
    std::vector<std::shared_ptr<IHandle>> const &handles, std::tuple<T...> &value) const {
    auto handle = std::static_pointer_cast<Handle<THead>>(handles.at(K));
    std::get<K>(value) = handle->Value();
    Unpacker<K + 1, TRest...>::Poke(handles, value);
}
}  // namespace inference
