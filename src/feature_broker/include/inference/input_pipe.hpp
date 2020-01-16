#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include "feature_broker_export.h"

namespace inference {
class TypeDescriptor;
class IHandle;
class ValueUpdater;
class OutputWaiter;

class InputPipe : public std::enable_shared_from_this<InputPipe> {
   public:
    virtual TypeDescriptor Type() const = 0;
    virtual ~InputPipe() = default;

   private:
    friend class FeatureBrokerBase;
    template <typename T>
    friend class DirectInputPipe;
    friend class OutputWaiterSinglePing;

    InputPipe() = default;

    class OutputWaiter final : public std::enable_shared_from_this<OutputWaiter> {
       public:
        FEATURE_BROKER_EXPORT OutputWaiter(size_t waiters);

        FEATURE_BROKER_EXPORT void Ping(bool subsequentCall);
        FEATURE_BROKER_EXPORT std::error_code Wait();
        FEATURE_BROKER_EXPORT bool Cleared();

       private:
        std::size_t _waiters;
        std::mutex _mutex;
        std::condition_variable _cv;
        bool _ready;
        bool _waiting{false};
    };

    virtual std::pair<std::shared_ptr<IHandle>, std::shared_ptr<ValueUpdater>> CreateHandleAndUpdater(
        std::shared_ptr<OutputWaiter> waiter) = 0;
};

}  // namespace inference
