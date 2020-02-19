// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cassert>
#include <rt/rt_expected.hpp>
#include <string>
#include <vector>
#include <vw_slim_model/actions.hpp>
#include <vw_slim_model/vw_error.hpp>

namespace vw_slim_model {
rt::expected<std::vector<int>> Actions::GetIntActions() {
    if (m_type != ActionType::Int) return make_vw_unexpected(vw_errc::invalid_actions);
    return std::static_pointer_cast<ActionHolder<int>>(m_actionHolder)->GetActions();
}

rt::expected<std::vector<std::string>> Actions::GetStringActions() {
    if (m_type != ActionType::String) return make_vw_unexpected(vw_errc::invalid_actions);
    return std::static_pointer_cast<ActionHolder<std::string>>(m_actionHolder)->GetActions();
}

rt::expected<std::vector<float>> Actions::GetFloatActions() {
    if (m_type != ActionType::Float) return make_vw_unexpected(vw_errc::invalid_actions);
    return std::static_pointer_cast<ActionHolder<float>>(m_actionHolder)->GetActions();
}

}  // namespace vw_slim_model
