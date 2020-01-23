#pragma once

#include <cstdint>
#include <string>
#include <system_error>
#include <tl/expected.hpp>

namespace vw_slim_model {
namespace priv {
enum class SchemaType : std::uint8_t { FloatString, FloatIndex, FloatsIndex, StringString, StringsString };

class SchemaEntry {
   public:
    SchemaEntry(SchemaEntry const& copy) noexcept;
    SchemaEntry(std::string const& inputName, std::string const& ns, const SchemaType type, const std::size_t index);
    SchemaEntry(std::string const& inputName, std::string const& ns, const SchemaType type, std::string const& name);

    const std::string InputName;
    const SchemaType Type;
    const std::string Namespace;
    const std::string Feature;
    const std::size_t Index;
};
}  // namespace priv
}  // namespace vw_slim_model