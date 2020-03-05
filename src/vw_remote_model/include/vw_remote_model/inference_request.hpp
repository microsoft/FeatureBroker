// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <memory>
#include <string>

namespace resonance_vw {

class InferenceRequest {
   public:
    virtual void GetResponse() = 0;
};
/*
class APSInferenceRequest : public InferenceRequest {
   public:
    static std::shared_ptr<InferenceRequest> CreateRecommendationRequest(
        std::map<std::string, std::string> const& featureMap, std::string const& baseUri,
        std::string const& experimentId) {
        return std::shared_ptr<InferenceRequest>(new APSInferenceRequest(featureMap, baseUri, experimentId));
    }

    void GetResponse() override;

   private:
    APSInferenceRequest(std::map<std::string, std::string> const& featureMap, std::string const& baseUri,
                        std::string const& experimentId);

    std::map<std::string, std::string> m_featureMap;
    std::string m_baseUri;
    std::string m_experimentId;
};
*/

}  // namespace resonance_vw
