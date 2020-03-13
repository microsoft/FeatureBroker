// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/input_pipe.hpp>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <tl/expected.hpp>
#include <vw_common/actions.hpp>

#include "vw_slim_model_export.h"

namespace resonance_vw {

/**
 * @brief An output task describes the outputs relative to the task of a VW model, for example, whether a model's
 * outputs are for a regression task, classification task, recommendation task, or other.
 */
class OutputTask {
   public:
    VW_SLIM_MODEL_EXPORT static std::shared_ptr<OutputTask> MakeRegression();

    VW_SLIM_MODEL_EXPORT static tl::expected<std::shared_ptr<OutputTask>, std::error_code> MakeRecommendation(
        std::shared_ptr<Actions> actions, std::string const& experimentId, std::string const& classNamesapce);

   private:
    OutputTask();
    virtual ~OutputTask();

    class IPoker {
       public:
        IPoker();
        virtual ~IPoker();
        virtual std::error_code Poke();
    };

    class IPokerImpl;

    virtual std::unordered_map<std::string, inference::TypeDescriptor> const& Outputs() const = 0;
    virtual tl::expected<OutputTask::IPoker*, std::error_code> CreatePoker(
        std::shared_ptr<void> vwModel, std::shared_ptr<void> vwExample,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe) const = 0;

    class Regression;
    class Recommendation;

    friend class Model;
    friend class ValueUpdater;
};
}  // namespace resonance_vw