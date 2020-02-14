// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <rt/rt_expected.hpp>
#include <string>
#include <vector>
#include <vw_slim_model/vw_error.hpp>

namespace vw_slim_model {

enum class ActionType : std::uint8_t { Float, String, Int, Unknown };

class Actions {
   public:
    template <typename T>
    static rt::expected<std::shared_ptr<Actions>> Create(std::vector<T> const& actions);
    ~Actions() {}

    VW_SLIM_MODEL_EXPORT rt::expected<std::vector<int>> GetIntActions();
    VW_SLIM_MODEL_EXPORT rt::expected<std::vector<std::string>> GetStringActions();
    VW_SLIM_MODEL_EXPORT rt::expected<std::vector<float>> GetFloatActions();

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

}  // namespace vw_slim_model

namespace vw_slim_model {
template <typename T>
inline rt::expected<std::shared_ptr<Actions>> Actions::Create(std::vector<T> const& actions) {
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

}  // namespace vw_slim_model
