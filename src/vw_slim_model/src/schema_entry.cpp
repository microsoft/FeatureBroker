#include "schema_entry.hpp"

#include <cassert>
#include <string>

namespace vw_slim_model {
namespace priv {

SchemaEntry::SchemaEntry(SchemaEntry const& copy) noexcept
    : InputName(copy.InputName), Type(copy.Type), Namespace(copy.Namespace), Feature(copy.Feature), Index(copy.Index) {}

SchemaEntry::SchemaEntry(std::string const& inputName, std::string const& ns, const SchemaType type,
                         const std::size_t index)
    : InputName(inputName), Namespace(ns), Type(type), Feature({}), Index(index) {
    assert(type == SchemaType::FloatIndex || type == SchemaType::FloatsIndex);
}

SchemaEntry::SchemaEntry(std::string const& inputName, std::string const& ns, const SchemaType type,
                         std::string const& name)
    : InputName(inputName), Namespace(ns), Type(type), Feature(name), Index(0) {
    assert(type == SchemaType::FloatString || type == SchemaType::StringsString || type == SchemaType::StringString);
}

}  // namespace priv
}  // namespace vw_slim_model