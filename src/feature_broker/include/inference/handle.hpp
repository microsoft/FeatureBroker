#pragma once

namespace inference {
class TypeDescriptor;

class IHandle {
   public:
    virtual ~IHandle() = default;

    virtual TypeDescriptor Type() const noexcept = 0;

    bool Changed() const noexcept { return _changed; }

   private:
    template <typename T>
    friend class Handle;
    template <typename T>
    friend class DirectInputPipe;
    template <typename T>
    friend class OutputPipe;  // Output pipe should not be changing handles.
    friend class TypeDescriptor;
    friend class FeatureBrokerBase;

    IHandle() = default;
    void Changed(bool changed) { _changed = changed; }
    bool _changed{false};
};

template <typename T>
class Handle final : public IHandle {
   public:
    Handle() = default;  // Ideally this should only be internally created anyway.
    virtual ~Handle() = default;

    TypeDescriptor Type() const noexcept override;
    T Value() const { return _value; }

   private:
    template <typename TT>
    friend class DirectInputPipe;
    template <typename TT>
    friend class OutputPipe;  // Output pipe should not be changing handles.
    friend class TypeDescriptor;
    friend class FeatureBrokerBase;

    T& MutableValue() { return _value; }

    T _value {};
};

}  // namespace inference

#include <inference/type_descriptor.hpp>
#include <tl/expected.hpp>

namespace inference {
template <typename T>
TypeDescriptor Handle<T>::Type() const noexcept {
    // By the time the feature broker has created the handle,
    // it would have validated this type.
    return TypeDescriptor::_CreateUnsafe<T>();
}

template <typename T>
rt::expected<std::shared_ptr<Handle<T>>> TryCast(std::shared_ptr<IHandle> handle) {
    auto typeExpected = TypeDescriptor::CreateExpected<T>();
    if (!typeExpected) return tl::make_unexpected(typeExpected.error());
    if (typeExpected.value() != handle->Type()) return make_feature_unexpected(feature_errc::type_mismatch);
    return std::static_pointer_cast<Handle<T>>(handle);
}
}  // namespace inference
