#include <inference/synchronous_feature_broker.hpp>

namespace inference {

SynchronousFeatureBroker::SynchronousFeatureBroker(std::shared_ptr<const Model> model) : FeatureBrokerBase(model) {}

}  // namespace inference
