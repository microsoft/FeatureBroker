// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>
#include <system_error>
#include <tl/expected.hpp>

#include "vw_common_export.h"

namespace resonance_vw {
namespace priv {
enum class SchemaType : std::uint8_t { FloatString, FloatIndex, FloatsIndex, StringString, StringsString };

class SchemaEntry {
   public:
    VW_COMMON_EXPORT SchemaEntry(SchemaEntry const& copy) noexcept;
    VW_COMMON_EXPORT SchemaEntry(std::string const& inputName, std::string const& ns, const SchemaType type,
                                 const std::size_t index);
    VW_COMMON_EXPORT SchemaEntry(std::string const& inputName, std::string const& ns, const SchemaType type,
                                 std::string const& name);

    const std::string InputName;
    const SchemaType Type;
    const std::string Namespace;
    const std::string Feature;
    const std::size_t Index;
};
}  // namespace priv
}  // namespace resonance_vw
