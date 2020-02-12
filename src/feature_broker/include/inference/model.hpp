// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <inference/handle.hpp>
#include <inference/input_pipe.hpp>
#include <inference/type_descriptor.hpp>
#include <inference/value_updater.hpp>
#include <map>
#include <memory>
#include <rt/rt_expected.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace inference {

class Model : public std::enable_shared_from_this<Model> {
   public:
    virtual ~Model() = default;
    virtual std::unordered_map<std::string, TypeDescriptor> const& Inputs() const = 0;
    virtual std::unordered_map<std::string, TypeDescriptor> const& Outputs() const = 0;
    virtual std::vector<std::string> GetRequirements(std::string const& outputName) const = 0;

    /**
     * @brief Create a Value Updater object.
     *
     * @param inputToHandle Map of input names to a handle holding input values out of which the outputs should be
     * calculated.
     * @param outputToPipe Map of output names to the pipes to which the value updater should push the inputs.
     * @param outOfBandNotifier In the event that a model has some sort of "non-input" value that can change its model,
     * this function should be called to indicate that the value has been changed. This could be, for example, due to a
     * change in the model state, or due to the model becoming "evaluable" when previously it was not. In cases where
     * there is no outside source "controlling" this, the method should just immediately call this function to indicate
     * that the model is "ready," but it is no longer necessary to do anything further with it, since beyond that point
     * it depends solely on whether the input changed or not only, and the library itself handles that.
     * @return An object that drives the computation from inputs to the outputs.
     */
    virtual rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<IHandle>> const& inputToHandle,
        std::map<std::string, std::shared_ptr<InputPipe>> const& outputToPipe,
        std::function<void()> outOfBandNotifier) const = 0;

   protected:
    Model() = default;
};

}  // namespace inference
