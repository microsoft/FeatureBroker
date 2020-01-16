#pragma once

#include "tl/expected.hpp"
#include <system_error>

namespace rt{
template <typename T>
using expected = tl::expected<T, std::error_code>;
}
