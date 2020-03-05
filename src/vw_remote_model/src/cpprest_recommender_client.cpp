// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cpprest/http_client.h>

#include <rt/rt_expected.hpp>
#include <vw_remote_model/cpprest_recommender_client.hpp>
#include <vw_remote_model/error.hpp>

using namespace utility;
using namespace web::http;
using namespace web::http::client;
using namespace concurrency::streams;

namespace resonance_vw {

// This is a fixed event id, but normally this should be generated for each request.
static string_t eventId(U("75269AD0-BFEE-4598-8196-C57383D38E10"));

rt::expected<std::shared_ptr<IRecommenderClient>> CppRestRecommenderClient::Create(
    std::string const& baseUri, std::string const& apsSubscriptionKey) {
    auto uri = utility::conversions::to_string_t(baseUri);
    auto key = utility::conversions::to_string_t(apsSubscriptionKey);
    if (uri.empty() || !uri::validate(uri)) {
        return make_remote_vw_unexpected(remote_vw_errc::invalid_url);
    }
    if (key.empty()) {
        return make_remote_vw_unexpected(remote_vw_errc::invalid_aps_subscription_key);
    }

    return std::shared_ptr<IRecommenderClient>(new CppRestRecommenderClient(uri, key));
}

CppRestRecommenderClient::CppRestRecommenderClient(string_t const& baseUri, string_t const& apsSubscriptionKey)
    : m_baseUri(baseUri), m_apsSubscriptionKey(apsSubscriptionKey) {
    // Use a proxy if one is set, this helps for debugging with Fiddler
    http_client_config config;
    config.set_proxy(web_proxy::use_auto_discovery);
    m_config = config;
}

rt::expected<std::string> CppRestRecommenderClient::GetRecommendation(
    std::unordered_map<std::string, std::string> const& features, std::vector<std::string> const& actions) {
    http_client client(m_baseUri, m_config);
    http_request request(methods::POST);

    // context features
    int index = 0;
    web::json::value jsonFeatures = web::json::value::array(features.size());
    for (auto feature : features) {
        web::json::value item = web::json::value::object();
        auto name = conversions::to_string_t(feature.first);
        auto value = conversions::to_string_t(feature.second);
        item[name] = web::json::value::string(value);
        jsonFeatures[index++] = item;
    }

    // actions
    index = 0;
    web::json::value jsonActions = web::json::value::array(actions.size());
    for (int i = 0; i < actions.size(); ++i) {
        web::json::value item = web::json::value::object();
        item[U("id")] = web::json::value::string(conversions::to_string_t(actions[i]));

        web::json::value itemArray = web::json::value::array();
        itemArray[0] = web::json::value::object();
        item[U("features")] = itemArray;
        jsonActions[index++] = item;
    }

    web::json::value body = web::json::value::object();
    body[U("contextFeatures")] = jsonFeatures;
    body[U("actions")] = jsonActions;
    body[U("excludedActions")] = web::json::value::array();
    body[U("eventId")] = web::json::value::string(eventId);
    body[U("deferActivation")] = web::json::value::boolean(false);

    request.headers().add(U("Ocp-Apim-Subscription-Key"), m_apsSubscriptionKey);
    request.headers().add(U("Content-Type"), U("application/json"));
    request.set_request_uri(U("/rank"));
    request.set_body(body);

    auto responseTask = client.request(request);
    auto actionIndex = 0;

    try {
        http_response response = responseTask.get();
        switch (response.status_code()) {
            case status_codes::BadRequest:
                return make_remote_vw_unexpected(remote_vw_errc::rank_request_invalid);
            case status_codes::Unauthorized:
                return make_remote_vw_unexpected(remote_vw_errc::rank_request_permission_denied);
            case status_codes::Created: {
                auto responseJson = response.extract_json().get();
                if (!responseJson.has_field(U("eventId")) || !responseJson.has_field(U("ranking"))) {
                    return make_remote_vw_unexpected(remote_vw_errc::rank_response_invalid);
                }

                // confirm the event id is the same.
                auto receivedEventId = responseJson.at(U("eventId")).as_string();
                if (receivedEventId != eventId) {
                    return make_remote_vw_unexpected(remote_vw_errc::rank_request_invalid_event_id);
                }

                auto rankings = responseJson[U("ranking")].as_array();
                auto& rank = *(rankings.begin());
                if (!rank.has_string_field(U("id"))) {
                    return make_remote_vw_unexpected(remote_vw_errc::rank_response_invalid);
                }

                actionIndex = std::stoi(rank.at(U("id")).as_string());
            } break;
            default:
                return make_remote_vw_unexpected(remote_vw_errc::unknown);
        }
    } catch (const http_exception& exc) {
        return tl::make_unexpected(exc.error_code());
    } catch (...) {
        return make_remote_vw_unexpected(remote_vw_errc::unknown);
    }

    if (actionIndex < 0 || actionIndex > actions.size() - 1) {
        return make_remote_vw_unexpected(remote_vw_errc::rank_response_invalid);
    }

    return actions[actionIndex];
}

}  // namespace resonance_vw
