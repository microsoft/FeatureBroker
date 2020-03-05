// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <map>
#include <rt/rt_expected.hpp>

namespace resonance_vw {

class IRecommenderClient {
   public:
    virtual rt::expected<std::string> GetRecommendation(std::unordered_map<std::string, std::string> const& features,
                                                        std::vector<std::string> const& actions) = 0;
};

}  // namespace resonance_vw
