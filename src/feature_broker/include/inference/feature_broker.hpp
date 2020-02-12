// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/direct_input_pipe.hpp>
#include <inference/feature_broker_base.hpp>
#include <inference/feature_error.hpp>
#include <inference/input_pipe.hpp>
#include <inference/model.hpp>
#include <inference/output_pipe_with_input.hpp>
#include <memory>
#include <string>
#include <system_error>
#include "feature_broker_export.h"
#include "rt/rt_expected.hpp"

namespace inference {
class FeatureBroker : public std::enable_shared_from_this<FeatureBroker>, FeatureBrokerBase {
   public:
    FEATURE_BROKER_EXPORT explicit FeatureBroker(std::shared_ptr<const Model> model = nullptr);
    FEATURE_BROKER_EXPORT virtual ~FeatureBroker();

    template <typename T>
    rt::expected<std::shared_ptr<DirectInputPipe<T>>> BindInput(std::string const &name) {
        return BindCore<DirectInputPipe<T>, typename DirectInputPipe<T>::Async, T>(name);
    }

    FEATURE_BROKER_EXPORT rt::expected<std::shared_ptr<FeatureBroker>> Fork(
        std::shared_ptr<const Model> model = nullptr) const;
    FEATURE_BROKER_EXPORT rt::expected<void> SetParent(std::shared_ptr<const FeatureBroker> newParent);

    using FeatureBrokerBase::BindInputs;

    using InputsType = FeatureBrokerBase::InputsType;

    template <typename T>
    rt::expected<std::shared_ptr<OutputPipeWithInput<T, InputsType>>> BindOutput(std::string const &name) {
        return FeatureBrokerBase::_BindOutput<T>(name);
    }

    template <class... T>
    rt::expected<std::shared_ptr<OutputPipeWithInput<std::tuple<T...>, InputsType>>> BindOutputs(
        const std::initializer_list<std::string> names) {
        return FeatureBrokerBase::_BindOutputs<T...>(names);
    }

   private:
    FEATURE_BROKER_EXPORT FeatureBroker(std::shared_ptr<const FeatureBroker> parent,
                                        std::shared_ptr<const Model> model);

    FEATURE_BROKER_EXPORT std::shared_ptr<const Model> GetModelOrNull(bool lock = true) const final override;
    FEATURE_BROKER_EXPORT std::shared_ptr<InputPipe> GetBindingOrNull(std::string const &name,
                                                                      bool lock = true) const final override;
    FEATURE_BROKER_EXPORT std::shared_ptr<FeatureProvider> GetProviderOrNull(std::string const &name,
                                                                             bool lock = true) const final override;

    // Have a private deleted cctor to avoid copying.
    FeatureBroker(const FeatureBroker &other) = delete;

    std::shared_ptr<const FeatureBroker> _parent;
};
}  // namespace inference
