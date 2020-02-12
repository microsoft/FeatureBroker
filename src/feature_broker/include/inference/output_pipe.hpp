// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/handle.hpp>
#include <rt/rt_expected.hpp>

namespace inference {

class IOutputPipe {
   public:
    virtual ~IOutputPipe() = default;
    virtual bool Changed() = 0;

   protected:
    IOutputPipe() = default;
};

template <typename T>
class OutputPipe : public IOutputPipe {
   public:
    virtual ~OutputPipe() = default;
    virtual rt::expected<bool> UpdateIfChanged(T& value) = 0;

   protected:
    OutputPipe() = default;
};

}  // namespace inference
