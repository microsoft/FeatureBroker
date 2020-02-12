// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

namespace inference {

template <typename T, class TInputs>
class OutputPipeWithInput : public OutputPipe<T> {
   public:
    OutputPipeWithInput() = default;
    virtual ~OutputPipeWithInput() = default;

    virtual const TInputs &Inputs() = 0;
    virtual rt::expected<void> WaitUntilChanged() = 0;
};

}  // namespace inference
