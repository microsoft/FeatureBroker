// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <vector>
#include <vw_common/error.hpp>
#include <vw_common/schema_builder.hpp>
#include <vw_common/schema_entry.hpp>

namespace resonance_vw {

typedef std::vector<priv::SchemaEntry> SchemaList;

SchemaBuilder::SchemaBuilder() : m_schema(std::make_shared<SchemaList>()) {}

SchemaBuilder::Expected SchemaBuilder::AddFloatFeature(std::string const& inputName, std::string const& featureName,
                                                       std::string const& ns) {
    if (!m_input_names.emplace(inputName).second) return make_vw_unexpected(vw_errc::duplicate_input_name);
    auto& schema = *std::static_pointer_cast<SchemaList>(m_schema);
    schema.push_back(priv::SchemaEntry(inputName, ns, priv::SchemaType::FloatString, featureName));
    return {};
}

SchemaBuilder::Expected SchemaBuilder::AddFloatFeature(std::string const& inputName, std::size_t offset,
                                                       std::string const& ns) {
    if (!m_input_names.emplace(inputName).second) return make_vw_unexpected(vw_errc::duplicate_input_name);
    auto& schema = *std::static_pointer_cast<SchemaList>(m_schema);
    schema.push_back(priv::SchemaEntry(inputName, ns, priv::SchemaType::FloatIndex, offset));
    return {};
}

SchemaBuilder::Expected SchemaBuilder::AddFloatVectorFeature(std::string const& inputName, std::size_t offset,
                                                             std::string const& ns) {
    if (!m_input_names.emplace(inputName).second) return make_vw_unexpected(vw_errc::duplicate_input_name);
    auto& schema = *std::static_pointer_cast<SchemaList>(m_schema);
    schema.push_back(priv::SchemaEntry(inputName, ns, priv::SchemaType::FloatsIndex, offset));
    return {};
}

VW_COMMON_EXPORT SchemaBuilder::Expected SchemaBuilder::AddIntFeature(std::string const& inputName, std::size_t offset,
                                                                      std::string const& ns) {
    if (!m_input_names.emplace(inputName).second) return make_vw_unexpected(vw_errc::duplicate_input_name);
    auto& schema = *std::static_pointer_cast<SchemaList>(m_schema);
    schema.push_back(priv::SchemaEntry(inputName, ns, priv::SchemaType::IntIndex, offset));
    return {};
}

SchemaBuilder::Expected SchemaBuilder::AddStringFeature(std::string const& inputName, std::string const& ns) {
    if (!m_input_names.emplace(inputName).second) return make_vw_unexpected(vw_errc::duplicate_input_name);
    auto& schema = *std::static_pointer_cast<SchemaList>(m_schema);
    schema.push_back(priv::SchemaEntry(inputName, ns, priv::SchemaType::StringString, 0));
    return {};
}

SchemaBuilder::Expected SchemaBuilder::AddStringVectorFeature(std::string const& inputName, std::string const& ns) {
    if (!m_input_names.emplace(inputName).second) return make_vw_unexpected(vw_errc::duplicate_input_name);
    auto& schema = *std::static_pointer_cast<SchemaList>(m_schema);
    schema.push_back(priv::SchemaEntry(inputName, ns, priv::SchemaType::StringString, 0));
    return {};
}

}  // namespace resonance_vw
