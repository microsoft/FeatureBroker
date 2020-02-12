// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

namespace inference {

template <typename T>
class Tensor final {
   public:
    Tensor() = default;
    Tensor(std::shared_ptr<T> ptr, std::vector<size_t> dimensions) : m_data(std::move(ptr)), m_dims(dimensions) {}
    Tensor(const Tensor<T>& other) : m_data(other.m_data), m_dims(other.m_dims) {}
    ~Tensor() = default;

    Tensor<T>& operator=(const Tensor<T>& other) {
        m_data = other.m_data;
        m_dims = other.m_dims;
        return *this;
    }

    T* Data() noexcept { return m_data.get(); }
    const T* Data() const noexcept { return m_data.get(); }
    std::vector<size_t> const& Dimensions() const noexcept { return m_dims; }

   private:
    std::shared_ptr<T> m_data;
    std::vector<size_t> m_dims;
};

}  // namespace inference
