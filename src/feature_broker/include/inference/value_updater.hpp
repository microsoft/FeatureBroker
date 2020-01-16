#pragma once

#include <string>
#include <system_error>

namespace inference {

class ValueUpdater {
   public:
    virtual ~ValueUpdater() = default;
    virtual bool Changed() { return true; }
    virtual std::error_code UpdateOutput() = 0;

   protected:
    ValueUpdater() = default;
};

}  // namespace inference
