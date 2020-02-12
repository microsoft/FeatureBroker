// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
//#define INFERENCE_USE_RTTI

#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#ifdef INFERENCE_USE_RTTI
#include <typeindex>
#endif
#include <rt/rt_expected.hpp>

#include "tensor.hpp"

namespace inference {

class InputPipe;

/**
 * Modeled after std::type_index but a more restricted set of types and not using RTTI.
 */
class TypeDescriptor final {
   public:
    template <typename T>
    static rt::expected<TypeDescriptor> CreateExpected() noexcept;

    // This non-std::error_code accepting convenience overload for those types that we are certain we always support.
    template <typename T>
    static TypeDescriptor Create() noexcept = delete;

    const static bool RuntimeTypesSupported =
#ifdef INFERENCE_USE_RTTI
        true;
#else
        false;
#endif

    bool operator==(const TypeDescriptor& rhs) const noexcept {
#ifdef INFERENCE_USE_RTTI
        return _typeIndex == rhs._typeIndex;
#else
        return _itemType == rhs._itemType && _containerType == rhs._containerType;
#endif
    }

    bool operator!=(const TypeDescriptor& rhs) const noexcept { return !(this->operator==(rhs)); }
    bool operator<(const TypeDescriptor& rhs) const noexcept {
#ifdef INFERENCE_USE_RTTI
        return _typeIndex < rhs._typeIndex;
#else
        return _itemType < rhs._itemType || (_itemType == rhs._itemType && _containerType < rhs._containerType);
#endif
    }
    bool operator<=(const TypeDescriptor& rhs) const noexcept {
#ifdef INFERENCE_USE_RTTI
        return _typeIndex <= rhs._typeIndex;
#else
        return _itemType < rhs._itemType || (_itemType == rhs._itemType && _containerType <= rhs._containerType);
#endif
    }
    bool operator>=(const TypeDescriptor& rhs) const noexcept { return !(this->operator<(rhs)); }

    std::size_t hash_code() const noexcept {
#ifdef INFERENCE_USE_RTTI
        return _typeIndex.hash_code();
#else
        return static_cast<std::size_t>(_itemType) +
               static_cast<std::size_t>(_containerType) * static_cast<std::size_t>(ItemType::_Limit);
#endif
    }

#ifdef INFERENCE_USE_RTTI
    bool IsUndefined() const noexcept { return false; }
#else
    bool IsUndefined() const noexcept { return _itemType == ItemType::Undefined; }
#endif

   private:
    template <typename T>
    friend class Handle;
    template <typename T>
    friend class DirectInputPipe;

    // It might be better if this was shared and pointing to some singleton, but maybe not -- the class itself is empty,
    // so basically an "instance" contains nothing but a virtual method table.

    template <typename T>
    inline static TypeDescriptor _CreateUnsafe() noexcept {
        auto td = TypeDescriptor::CreateExpected<T>();
        if (td) return td.value();
        return TypeDescriptor(ItemType::Undefined, ContainerType::Scalar, new TypeServices<T>());
    }

#ifdef INFERENCE_USE_RTTI
    const std::type_index _typeIndex;
#else
    enum class ItemType : int {
        Undefined,
        Single,
        Double,
        Int,
        Long,
        String,
        _Limit  // Should be the last one, so that hashing and range tests can work.
    };

    enum class ContainerType : int { Scalar, Tensor };

    const ItemType _itemType;
    const ContainerType _containerType;
#endif

    class ITypeServices {
       public:
        virtual std::shared_ptr<InputPipe> CreateDirectInputPipeSyncSingleConsumer() const noexcept = 0;
        virtual ~ITypeServices() = default;
    };

    // There is no reason for this to be shared. I *think* we could get away with using a unique ptr if we were explicit
    // about copy or more constructors?
    const std::shared_ptr<ITypeServices> _typeServices;

#ifdef INFERENCE_USE_RTTI
    TypeDescriptor(const std::type_index typeIndex, ITypeServices* services)
        : _typeIndex(typeIndex), _typeServices(services) {}
#else
    TypeDescriptor(const ItemType itemType, const ContainerType containerType, ITypeServices* services)
        : _itemType(itemType), _containerType(containerType), _typeServices(services) {}
#endif

    // Type services are type-specialized things that are used by the feature broker level, to support the
    // infrastructure.

    template <typename T>
    class TypeServices final : public ITypeServices {
       public:
        TypeServices() = default;
        virtual ~TypeServices() = default;
        std::shared_ptr<InputPipe> CreateDirectInputPipeSyncSingleConsumer() const noexcept;
    };

    friend class FeatureBrokerBase;

    std::shared_ptr<InputPipe> CreateDirectInputPipeSyncSingleConsumer() const noexcept {
        return _typeServices->CreateDirectInputPipeSyncSingleConsumer();
    }
};

}  // namespace inference

#include <inference/direct_input_pipe.hpp>
#include <inference/feature_error.hpp>

namespace inference {
template <typename T>
rt::expected<TypeDescriptor> TypeDescriptor::CreateExpected() noexcept {
#ifdef INFERENCE_USE_RTTI
    return TypeDescriptor(typeid(T), new TypeServices<T>());
#else
    const auto scalar = ContainerType::Scalar;
    if (std::is_same<T, int32_t>()) return TypeDescriptor(ItemType::Int, scalar, new TypeServices<T>());
    if (std::is_same<T, int64_t>()) return TypeDescriptor(ItemType::Long, scalar, new TypeServices<T>());
    if (std::is_same<T, float>()) return TypeDescriptor(ItemType::Single, scalar, new TypeServices<T>());
    if (std::is_same<T, double>()) return TypeDescriptor(ItemType::Double, scalar, new TypeServices<T>());
    if (std::is_same<T, std::string>()) return TypeDescriptor(ItemType::String, scalar, new TypeServices<T>());
    const auto tensor = ContainerType::Tensor;
    if (std::is_same<T, Tensor<int32_t>>()) return TypeDescriptor(ItemType::Int, tensor, new TypeServices<T>());
    if (std::is_same<T, Tensor<int64_t>>()) return TypeDescriptor(ItemType::Long, tensor, new TypeServices<T>());
    if (std::is_same<T, Tensor<float>>()) return TypeDescriptor(ItemType::Single, tensor, new TypeServices<T>());
    if (std::is_same<T, Tensor<double>>()) return TypeDescriptor(ItemType::Double, tensor, new TypeServices<T>());
    if (std::is_same<T, Tensor<std::string>>()) return TypeDescriptor(ItemType::String, tensor, new TypeServices<T>());
    //    return TypeDescriptor(ItemType::Single, ContainerType::Tensor);

    return make_feature_unexpected(feature_errc::type_unsupported);
#endif
}

template <typename T>
std::shared_ptr<InputPipe> TypeDescriptor::TypeServices<T>::CreateDirectInputPipeSyncSingleConsumer() const noexcept {
    return std::static_pointer_cast<InputPipe>(std::make_shared<typename DirectInputPipe<T>::SyncSingleConsumer>());
}

/**
 * @return A type descriptor for int32_t.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<int32_t>() noexcept {
    // You could imagine a more direct version of these that returns a singleton for the common types rather than
    // creating a new one.
    return _CreateUnsafe<int32_t>();
}

/**
 * @return A type descriptor for int64_t.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<int64_t>() noexcept {
    return _CreateUnsafe<int64_t>();
}

/**
 * @return A type descriptor for single precision floating point.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<float>() noexcept {
    return _CreateUnsafe<float>();
}

/**
 * @return A type descriptor for double precision floating point.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<double>() noexcept {
    return _CreateUnsafe<double>();
}

/**
 * @return A type descriptor for strings.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<std::string>() noexcept {
    return _CreateUnsafe<std::string>();
}

/**
 * @return A type descriptor for int32_t.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<Tensor<int32_t>>() noexcept {
    // You could imagine a more direct version of these that returns a singleton for the common types rather than
    // creating a new one.
    return _CreateUnsafe<Tensor<int32_t>>();
}

/**
 * @return A type descriptor for int64_t.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<Tensor<int64_t>>() noexcept {
    return _CreateUnsafe<Tensor<int64_t>>();
}

/**
 * @return A type descriptor for single precision floating point.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<Tensor<float>>() noexcept {
    return _CreateUnsafe<Tensor<float>>();
}

/**
 * @return A type descriptor for double precision floating point.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<Tensor<double>>() noexcept {
    return _CreateUnsafe<Tensor<double>>();
}

/**
 * @return A type descriptor for strings.
 */
template <>
inline TypeDescriptor TypeDescriptor::Create<Tensor<std::string>>() noexcept {
    return _CreateUnsafe<Tensor<std::string>>();
}

}  // namespace inference
