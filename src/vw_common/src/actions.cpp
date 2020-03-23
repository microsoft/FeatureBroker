// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cassert>
#include <string>
#include <vector>
#include <vw_common/actions.hpp>
#include <vw_common/error.hpp>

namespace resonance_vw {
Actions::IntExpected Actions::GetIntActions() const {
    if (m_type != ActionType::Int) return make_vw_unexpected(vw_errc::invalid_actions);
    return std::static_pointer_cast<ActionHolder<int>>(m_actionHolder)->GetActions();
}

Actions::StringExpected Actions::GetStringActions() const {
    if (m_type != ActionType::String) return make_vw_unexpected(vw_errc::invalid_actions);
    return std::static_pointer_cast<ActionHolder<std::string>>(m_actionHolder)->GetActions();
}

Actions::FloatExpected Actions::GetFloatActions() const {
    if (m_type != ActionType::Float) return make_vw_unexpected(vw_errc::invalid_actions);
    return std::static_pointer_cast<ActionHolder<float>>(m_actionHolder)->GetActions();
}

}  // namespace resonance_vw
