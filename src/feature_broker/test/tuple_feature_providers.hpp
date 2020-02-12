// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <inference/direct_input_pipe.hpp>
#include <inference/feature_error.hpp>
#include <inference/feature_provider.hpp>
#include <inference/type_descriptor.hpp>
#include <inference/value_updater.hpp>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ::inference;

namespace inference_test {
// A flexible simple feature provider based on std::tuples.
template <class... T>
class TupleProvider : public FeatureProvider {
   public:
    explicit TupleProvider(std::vector<std::pair<std::string, TypeDescriptor>> const &outputs)
        : _lastUpdate(outputs.size()) {
        std::size_t index = 0;
        for (auto &pair : outputs) {
            _outputs.emplace(pair.first, pair.second);
            _indices.emplace(pair.first, index++);
        }
    }

    std::unordered_map<std::string, TypeDescriptor> const &Outputs() const override { return _outputs; }

    template <size_t K, class TItem>
    void Set(TItem value);

    rt::expected<std::shared_ptr<ValueUpdater>> CreateValueUpdater(
        std::map<std::string, std::shared_ptr<InputPipe>> const &outputToPipe,
        std::function<void()> valuesChangedNotifier) const override;

   private:
    friend class Updater;
    mutable std::mutex _updatersMutex;

    std::tuple<T...> _values;
    std::vector<std::size_t> _lastUpdate;
    std::unordered_map<std::string, TypeDescriptor> _outputs;
    std::unordered_map<std::string, std::size_t> _indices;

    class Updater : public ValueUpdater {
       public:
        std::error_code error;

        Updater(std::shared_ptr<const TupleProvider> parent,
                std::map<std::string, std::shared_ptr<InputPipe>> const &outputToPipe, std::function<void()> pinger)
            : _pinger(pinger),
              _parent(std::move(parent)),
              _pipes(_parent->_indices.size()),
              _lastUpdate(_parent->_indices.size()) {
            bool anyUpdated = false;
            for (const auto &pair : outputToPipe) {
                std::size_t index = _parent->_indices.at(pair.first);
                _pipes[index] = pair.second;
                if (_parent->_lastUpdate[index] > 0) anyUpdated = true;
            }
            if (anyUpdated) _pinger();
        }

        bool ActiveOn(size_t index) const noexcept { return _pipes.at(index) != nullptr; }
        void Ping() { _pinger(); }

        bool Changed() override {
            for (std::size_t i = 0; i < _pipes.size(); ++i) {
                if (_pipes[i] && (_lastUpdate[i] < _parent->_lastUpdate[i])) return true;
            }
            return false;
        }

        std::error_code UpdateOutput() override {
            _unpacker.Feed(*this);
            return std::error_code();
        }

       private:
        const std::function<void()> _pinger;
        const std::shared_ptr<const TupleProvider> _parent;
        std::vector<std::shared_ptr<InputPipe>> _pipes;
        std::vector<std::size_t> _lastUpdate;

        // general specification.
        template <size_t K, class... TTuple>
        struct Unpacker {
            Unpacker() = delete;
        };

        // Base case, with no args left.
        template <size_t K>
        struct Unpacker<K> {
            void Feed(Updater &updater) {}
        };

        // Recursive case.
        template <size_t K, class THead, class... TRest>
        struct Unpacker<K, THead, TRest...> : Unpacker<K + 1, TRest...> {
            void Feed(Updater &updater) {
                auto pipe = std::static_pointer_cast<DirectInputPipe<THead>>(updater._pipes.at(K));
                if (pipe && updater._lastUpdate.at(K) < updater._parent->_lastUpdate.at(K)) {
                    updater._lastUpdate[K] = updater._parent->_lastUpdate.at(K);
                    pipe->Feed(std::get<K>(updater._parent->_values));
                }
                Unpacker<K + 1, TRest...>::Feed(updater);
            }
        };

        Unpacker<0, T...> _unpacker;
    };

    mutable std::vector<std::weak_ptr<Updater>> _updaters;
};

class TupleProviderFactory {
   public:
    // This could be done with variadic templates, but I can't really be bothered for test code where the number we'd
    // need to support is quite finite...
    template <class T1>
    static std::shared_ptr<TupleProvider<T1>> Create(std::string name1) {
        std::vector<std::pair<std::string, TypeDescriptor>> outputs = {{name1, TypeDescriptor::Create<T1>()}};
        return std::make_shared<TupleProvider<T1>>(outputs);
    }

    template <class T1, class T2>
    static std::shared_ptr<TupleProvider<T1, T2>> Create(std::string name1, std::string name2) {
        std::vector<std::pair<std::string, TypeDescriptor>> outputs = {{name1, TypeDescriptor::Create<T1>()},
                                                                       {name2, TypeDescriptor::Create<T2>()}};
        return std::make_shared<TupleProvider<T1, T2>>(outputs);
    }

    template <class T1, class T2, class T3>
    static std::shared_ptr<TupleProvider<T1, T2, T3>> Create(std::string name1, std::string name2, std::string name3) {
        std::vector<std::pair<std::string, TypeDescriptor>> outputs = {{name1, TypeDescriptor::Create<T1>()},
                                                                       {name2, TypeDescriptor::Create<T2>()},
                                                                       {name3, TypeDescriptor::Create<T3>()}};
        return std::make_shared<TupleProvider<T1, T2, T3>>(outputs);
    }

   private:
    TupleProviderFactory() = delete;
};  // namespace inference_test

template <class... T>
template <size_t K, class TItem>
void TupleProvider<T...>::Set(TItem value) {
    std::get<K>(_values) = value;
    ++_lastUpdate[K];

    std::unique_lock<std::mutex> lock(_updatersMutex);
    for (auto iter = _updaters.begin(); iter != _updaters.end();) {
        if (auto updater = iter->lock()) {
            if (updater->ActiveOn(K)) updater->Ping();
            ++iter;
        } else {
            iter = _updaters.erase(iter);
        }
    }
}

template <class... T>
rt::expected<std::shared_ptr<ValueUpdater>> TupleProvider<T...>::CreateValueUpdater(
    std::map<std::string, std::shared_ptr<InputPipe>> const &outputToPipe,
    std::function<void()> valuesChangedNotifier) const {
    auto thisPtr = std::static_pointer_cast<const TupleProvider<T...>>(shared_from_this());
    auto updater = std::make_shared<Updater>(thisPtr, outputToPipe, valuesChangedNotifier);
    if (updater->error) return tl::make_unexpected(updater->error);
    std::unique_lock<std::mutex> lock(_updatersMutex);
    _updaters.push_back(updater);
    return std::static_pointer_cast<ValueUpdater>(updater);
}

}  // namespace inference_test
