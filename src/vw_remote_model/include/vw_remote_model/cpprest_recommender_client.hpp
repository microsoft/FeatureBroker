// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cpprest/http_client.h>

#include <map>
#include <rt/rt_expected.hpp>
#include <string>
#include <vector>
#include <vw_remote_model/irecommender_client.hpp>

#include "vw_remote_model_export.h"

namespace resonance_vw {
class CppRestRecommenderClient : public IRecommenderClient {
   public:
    VW_REMOTE_MODEL_EXPORT static rt::expected<std::shared_ptr<IRecommenderClient>> Create(
        std::string const& baseUri, std::string const& apsSubscriptionKey);

    VW_REMOTE_MODEL_EXPORT virtual rt::expected<std::string> GetRecommendation(
        std::unordered_map<std::string, std::string> const& features, std::vector<std::string> const& actions) override;

   protected:
    CppRestRecommenderClient(utility::string_t const& baseUri, utility::string_t const& apsSubscriptionKey);

   private:
    utility::string_t m_baseUri;
    utility::string_t m_apsSubscriptionKey;
    web::http::client::http_client_config m_config;
};
}  // namespace resonance_vw
