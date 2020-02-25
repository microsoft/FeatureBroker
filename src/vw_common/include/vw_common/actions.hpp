// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tl/expected.hpp>
#include <vector>
#include <vw_common/error.hpp>

#include "vw_common_export.h"

namespace resonance_vw {

enum class ActionType : std::uint8_t { Float, String, Int, Unknown };

class Actions {
   public:
    using Expected = tl::expected<std::shared_ptr<Actions>, std::error_code>;
    using IntExpected = tl::expected<std::vector<int>, std::error_code>;
    using StringExpected = tl::expected<std::vector<std::string>, std::error_code>;
    using FloatExpected = tl::expected<std::vector<float>, std::error_code>;

    template <typename T>
    static Expected Create(std::vector<T> const& actions);
    ~Actions() {}

    VW_COMMON_EXPORT IntExpected GetIntActions();
    VW_COMMON_EXPORT StringExpected GetStringActions();
    VW_COMMON_EXPORT FloatExpected GetFloatActions();

    ActionType Type() const { return m_type; }

   private:
    Actions() = default;
    ActionType m_type;
    std::shared_ptr<void> m_actionHolder;

    template <typename T>
    class ActionHolder {
       public:
        ActionHolder(std::vector<T> const& actions) : m_actions(actions) {}
        ~ActionHolder() = default;

        std::vector<T> GetActions() const { return m_actions; }

       private:
        ActionHolder() = default;
        std::vector<T> m_actions;
    };

    template <typename T>
    static ActionType FromType();
};

}  // namespace resonance_vw

namespace resonance_vw {
template <typename T>
inline Actions::Expected Actions::Create(std::vector<T> const& actions) {
    if (Actions::FromType<T>() == ActionType::Unknown) return make_vw_unexpected(vw_errc::invalid_actions);
    if (actions.size() == 0) return make_vw_unexpected(vw_errc::invalid_actions);

    std::shared_ptr<Actions> value(new Actions());
    value->m_actionHolder = std::make_shared<ActionHolder<T>>(actions);
    value->m_type = Actions::FromType<T>();
    return value;
}

template <typename T>
inline ActionType Actions::FromType() {
    return ActionType::Unknown;
}

template <>
inline ActionType Actions::FromType<float>() {
    return ActionType::Float;
}

template <>
inline ActionType Actions::FromType<int>() {
    return ActionType::Int;
}

template <>
inline ActionType Actions::FromType<std::string>() {
    return ActionType::String;
}

}  // namespace resonance_vw
