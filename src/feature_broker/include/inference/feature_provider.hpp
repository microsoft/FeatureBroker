#pragma once

#include <inference/input_pipe.hpp>
#include <inference/type_descriptor.hpp>
#include <inference/value_updater.hpp>
#include <map>
#include <memory>
#include <rt/rt_expected.hpp>
#include <string>
#include <unordered_map>

namespace inference {

class FeatureProvider : public std::enable_shared_from_this<FeatureProvider> {
   public:
    virtual ~FeatureProvider() = default;
    virtual std::unordered_map<std::string, TypeDescriptor> const& Outputs() const = 0;

    /**
     * @brief Create a Value Updater object.
     *
     * @param outputToPipe Map of output names to the pipes to which the value updater should push the inputs. This will
     * be a subset of the outputs.
     * @param valuesChangedNotifier The value updater should have this method changed when any of the values it is
     * publishing have occassion to change.
     * @return An object that drives pushing values to the outputs.
     */
    virtual rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<InputPipe>> const& outputToPipe,
        std::function<void()> valuesChangedNotifier) const = 0;

   protected:
    FeatureProvider() = default;
};

}  // namespace inference
