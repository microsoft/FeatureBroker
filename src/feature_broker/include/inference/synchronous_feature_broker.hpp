// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/direct_input_pipe.hpp>
#include <inference/feature_broker_base.hpp>
#include <inference/feature_error.hpp>
#include <inference/input_pipe.hpp>
#include <inference/model.hpp>
#include <inference/output_pipe.hpp>
#include <memory>
#include <string>
#include <system_error>
#include "feature_broker_export.h"

namespace inference {
class SynchronousFeatureBroker : FeatureBrokerBase {
   public:
    FEATURE_BROKER_EXPORT explicit SynchronousFeatureBroker(std::shared_ptr<const Model> model);
    virtual ~SynchronousFeatureBroker() = default;

    template <typename T>
    rt::expected<std::shared_ptr<DirectInputPipe<T>>> BindInput(std::string const &name) {
        return BindCore<DirectInputPipe<T>, typename DirectInputPipe<T>::SyncSingleConsumer, T>(name);
    }

    using FeatureBrokerBase::BindInputs;

    template <typename T>
    rt::expected<std::shared_ptr<OutputPipe<T>>> BindOutput(std::string const &name) {
        auto result = FeatureBrokerBase::_BindOutput<T>(name);
        if (result) return std::static_pointer_cast<OutputPipe<T>>(result.value());
        return tl::make_unexpected(result.error());
    }

    template <class... T>
    rt::expected<std::shared_ptr<OutputPipe<std::tuple<T...>>>> BindOutputs(
        const std::initializer_list<std::string> names) {
        auto result = FeatureBrokerBase::_BindOutputs<T...>(names);
        if (result) return std::static_pointer_cast<OutputPipe<std::tuple<T...>>>(result.value());
        return result.error();
    }

   private:
    // Have a private deleted cctor to avoid copying.
    SynchronousFeatureBroker(const SynchronousFeatureBroker &other) = delete;
};
}  // namespace inference
