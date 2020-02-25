// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef _WIN32
// There appear to be some includes missing from memory.h that let it compile.
// Also, VW uses throw within memory.h despite VW_NOEXCEPT being set for some
// reason, and this only fails on Linux because the requisite defines needed to
// trigger the compilation of that code are not defined in Windows.
#include <unistd.h>  // So memory.h has access to _SC_PAGE_SIZE.
#define THROW        // So VW's memory.h doesn't fail when THROW is encountered. This seems subideal.
#endif

#include <unordered_map>
#include <vw_common/actions.hpp>
#include <vw_common/schema_builder.hpp>
#include <vw_common/schema_entry.hpp>
#include <vw_slim_model/model.hpp>

// There are unfortunately more than a few compiler warnings in the VW Slim headers. For the sake of being aware of
// warnings in thi codebase without having them be lost in the flood of VW warnings, we have some targeted disabling of
// warnings.
#pragma warning(push)
#pragma warning(disable : 26451)  // Arithmetic overflow warnings in hash.h.
#pragma warning(disable : 26495)  // Uninitialized members and variables seemingly everywhere.
#pragma warning(disable : 26812)  // Enums used throughout instad of enum class.
#include "array_parameters.h"
#include "example_predict_builder.h"
#include "vw_slim_predict.h"
#include "vw_slim_return_codes.h"
#pragma warning(pop)

namespace resonance_vw {

typedef std::vector<priv::SchemaEntry> SchemaList;
typedef vw_slim::vw_predict<::sparse_parameters> VWModel;

class Model::State final {
   public:
    State(SchemaBuilder const& schemaBuilder, std::shared_ptr<VWModel> vwModel);

    const SchemaList m_schema;
    const std::shared_ptr<VWModel> m_model;

    std::vector<size_t> m_schema_entry_idx_to_idx;
    std::vector<size_t> m_builder_idx_to_idx;
};

Model::State::State(SchemaBuilder const& schemaBuilder, std::shared_ptr<VWModel> vwModel)
    : m_schema(*std::static_pointer_cast<SchemaList>(schemaBuilder.Schema())), m_model(std::move(vwModel)) {
    std::unordered_map<std::string, size_t> entryIdxToBuilderIdx;
    m_schema_entry_idx_to_idx.reserve(m_schema.size());
    for (size_t i = 0; i < m_schema.size(); ++i) {
        auto emplacement = entryIdxToBuilderIdx.emplace(m_schema[i].Namespace, entryIdxToBuilderIdx.size());
        if (emplacement.second) m_builder_idx_to_idx.push_back(i);
        m_schema_entry_idx_to_idx.push_back(emplacement.first->second);
    }
}

class ValueUpdaterBase : public inference::ValueUpdater {
   public:
    class IPeeker {
       public:
        virtual ~IPeeker() = default;
        virtual std::error_code Peek(vw_slim::example_predict_builder& builder) = 0;

       protected:
        IPeeker() = default;
    };

    ValueUpdaterBase(std::shared_ptr<Model::State> state, std::shared_ptr<::safe_example_predict> example,
                     std::vector<std::unique_ptr<ValueUpdaterBase::IPeeker>>&& peekers);
    std::error_code DoPeeks();

    static std::unique_ptr<IPeeker> CreatePeeker(priv::SchemaEntry const& entry,
                                                 std::shared_ptr<vw_slim::example_predict_builder> ex,
                                                 std::shared_ptr<inference::IHandle> handle);

    template <typename T>
    class Peeker : public IPeeker {
       public:
        virtual ~Peeker() = default;

       protected:
        const std::shared_ptr<inference::Handle<T>> m_handle;
        Peeker(std::shared_ptr<inference::IHandle> handle);
    };

   protected:
    const std::shared_ptr<Model::State> m_state;
    const std::shared_ptr<::safe_example_predict> m_vw_example;
    const std::vector<std::unique_ptr<IPeeker>> m_peekers;
    std::vector<std::unique_ptr<vw_slim::example_predict_builder>> m_builders;
};

class ValueUpdaterFloat final : public ValueUpdaterBase {
   public:
    ValueUpdaterFloat(std::shared_ptr<Model::State> state, std::shared_ptr<::safe_example_predict> example,
                      std::vector<std::unique_ptr<ValueUpdaterBase::IPeeker>>&& peekers,
                      std::shared_ptr<inference::DirectInputPipe<float>> pipe);
    std::error_code UpdateOutput() override;

   private:
    const std::shared_ptr<inference::DirectInputPipe<float>> m_pipe;
};

template <typename T>
ValueUpdaterBase::Peeker<T>::Peeker(std::shared_ptr<inference::IHandle> handle)
    : m_handle(std::static_pointer_cast<inference::Handle<T>>(handle)) {}

class FloatIndexPeeker final : public ValueUpdaterBase::Peeker<float> {
   public:
    std::error_code Peek(vw_slim::example_predict_builder& builder) {
        builder.push_feature(m_index, m_handle->Value());
        return {};
    }
    FloatIndexPeeker(priv::SchemaEntry const& entry, std::shared_ptr<inference::IHandle> handle)
        : ValueUpdaterBase::Peeker<float>(handle), m_index(entry.Index) {}

   private:
    const decltype(priv::SchemaEntry::Index) m_index;
};

class FloatStringPeeker final : public ValueUpdaterBase::Peeker<float> {
   public:
    std::error_code Peek(vw_slim::example_predict_builder& builder) {
        // Ewwww. The VW code doesn't actually modify this string at all, it just Murmur hashes it, so in principle they
        // could have put a const qualifier on the method argument, but they didn't, so, here we are. Perhaps we could
        // change their code to add const and other modifiers were possible, and remove this cast in future versions.
        builder.push_feature_string((char*)(m_index.c_str()), m_handle->Value());
        return {};
    }
    FloatStringPeeker(priv::SchemaEntry const& entry, std::shared_ptr<inference::IHandle> handle)
        : ValueUpdaterBase::Peeker<float>(handle), m_index(entry.Feature) {}

   private:
    const decltype(priv::SchemaEntry::Feature) m_index;
};

class FloatsIndexPeeker final : public ValueUpdaterBase::Peeker<inference::Tensor<float>> {
   public:
    std::error_code Peek(vw_slim::example_predict_builder& builder) {
        auto val = m_handle->Value();
        auto pdata = val.Data();
        size_t total = 1;
        for (const auto d : val.Dimensions()) total *= d;
        for (size_t i = 0; i < total; ++i) {
            // This test will also result in NaN being skipped. Is that desirable/undesirable?
            if (pdata[i] != 0) builder.push_feature(i + m_index, pdata[i]);
        }
        return {};
    }
    FloatsIndexPeeker(priv::SchemaEntry const& entry, std::shared_ptr<inference::IHandle> handle)
        : ValueUpdaterBase::Peeker<inference::Tensor<float>>(handle), m_index(entry.Index) {}

   private:
    const decltype(priv::SchemaEntry::Index) m_index;
};

class StringStringPeeker final : public ValueUpdaterBase::Peeker<std::string> {
   public:
    std::error_code Peek(vw_slim::example_predict_builder& builder) {
        // Again eww.
        builder.push_feature_string((char*)(m_handle->Value().c_str()), 1.0f);
        return {};
    }
    StringStringPeeker(std::shared_ptr<inference::IHandle> handle) : ValueUpdaterBase::Peeker<std::string>(handle) {}
};

class StringsStringPeeker final : public ValueUpdaterBase::Peeker<inference::Tensor<std::string>> {
   public:
    std::error_code Peek(vw_slim::example_predict_builder& builder) {
        auto val = m_handle->Value();
        auto pdata = val.Data();
        size_t total = 1;
        for (const auto d : val.Dimensions()) total *= d;
        for (size_t i = 0; i < total; ++i) {
            // This test will also result in NaN being skipped. Is that desirable/undesirable?
            builder.push_feature_string((char*)(pdata[i].c_str()), 1.0f);
        }
        return {};
    }
    StringsStringPeeker(std::shared_ptr<inference::IHandle> handle)
        : ValueUpdaterBase::Peeker<inference::Tensor<std::string>>(handle) {}
};

std::unique_ptr<ValueUpdaterBase::IPeeker> ValueUpdaterBase::CreatePeeker(
    priv::SchemaEntry const& entry, std::shared_ptr<vw_slim::example_predict_builder> ex,
    std::shared_ptr<inference::IHandle> handle) {
    switch (entry.Type) {
        case priv::SchemaType::FloatIndex:
            return std::make_unique<FloatIndexPeeker>(entry, handle);
        case priv::SchemaType::FloatString:
            return std::make_unique<FloatStringPeeker>(entry, handle);
        case priv::SchemaType::FloatsIndex:
            return std::make_unique<FloatsIndexPeeker>(entry, handle);
        case priv::SchemaType::StringString:
            return std::make_unique<StringStringPeeker>(handle);
        default:
            assert(entry.Type == priv::SchemaType::StringsString);
            return std::make_unique<StringsStringPeeker>(handle);
    }
}

// ValueUpdater constructors ----------------------------------------

ValueUpdaterBase::ValueUpdaterBase(std::shared_ptr<Model::State> state, std::shared_ptr<::safe_example_predict> example,
                                   std::vector<std::unique_ptr<ValueUpdaterBase::IPeeker>>&& peekers)
    : m_state(std::move(state)),
      m_vw_example(std::move(example)),
      m_peekers(std::move(peekers)),
      m_builders(m_state->m_builder_idx_to_idx.size()) {}

ValueUpdaterFloat::ValueUpdaterFloat(std::shared_ptr<Model::State> state,
                                     std::shared_ptr<::safe_example_predict> example,
                                     std::vector<std::unique_ptr<ValueUpdaterBase::IPeeker>>&& peekers,
                                     std::shared_ptr<inference::DirectInputPipe<float>> pipe)
    : ValueUpdaterBase(std::move(state), std::move(example), std::move(peekers)), m_pipe(std::move(pipe)) {}

// ValueUpdater peeking and updating ----------------------------------------

std::error_code ValueUpdaterBase::DoPeeks() {
    m_vw_example->clear();
    // Set up the new round of builders. This seems inefficient, but I see no way to reuse builders.
    const auto& bi2ei = m_state->m_builder_idx_to_idx;
    for (size_t i = 0; i < bi2ei.size(); ++i) {
        const auto& entry = m_state->m_schema[bi2ei[i]];
        m_builders[i] =
            std::make_unique<vw_slim::example_predict_builder>(m_vw_example.get(), (char*)entry.Namespace.c_str());
    }

    const auto& ei2bi = m_state->m_schema_entry_idx_to_idx;
    for (size_t i = 0; i < ei2bi.size(); ++i) {
        std::error_code result = m_peekers[i]->Peek(*m_builders[ei2bi[i]]);
        if (result) return result;
    }
    return {};
}

std::error_code ValueUpdaterFloat::UpdateOutput() {
    DoPeeks();
    float result = 0;
    int err = m_state->m_model->predict(*m_vw_example, result);
    if (err) return make_vw_error(vw_errc::predict_failure);
    m_pipe->Feed(result);
    return {};
}

rt::expected<std::shared_ptr<Model>> Model::Load(SchemaBuilder const& schemaBuilder,
                                                 std::vector<char> const& modelBytes) {
    auto model = std::make_shared<VWModel>();
    auto code = model->load(modelBytes.data(), modelBytes.size());
    if (code != S_VW_PREDICT_OK) return make_vw_unexpected(vw_errc::load_failure);
    return std::shared_ptr<Model>(new Model(schemaBuilder, std::static_pointer_cast<void>(model)));
}
rt::expected<std::shared_ptr<Model>> Model::Load(SchemaBuilder const& schemaBuilder, Actions const& actions,
                                                 std::vector<char> const& modelBytes) {
    auto model = std::make_shared<VWModel>();
    auto code = model->load(modelBytes.data(), modelBytes.size());
    if (code != S_VW_PREDICT_OK) return make_vw_unexpected(vw_errc::load_failure);
    return std::shared_ptr<Model>(new Model(schemaBuilder, std::static_pointer_cast<void>(model)));
}

inference::TypeDescriptor EmplaceInputEntry(priv::SchemaEntry const& entry) noexcept {
    switch (entry.Type) {
        case priv::SchemaType::FloatIndex:
        case priv::SchemaType::FloatString:
            return inference::TypeDescriptor::Create<float>();
        case priv::SchemaType::FloatsIndex:
            return inference::TypeDescriptor::Create<inference::Tensor<float>>();
        case priv::SchemaType::StringString:
            return inference::TypeDescriptor::Create<std::string>();
        default:
            assert(entry.Type == priv::SchemaType::StringsString);
            return inference::TypeDescriptor::Create<inference::Tensor<std::string>>();
    }
}

Model::Model(SchemaBuilder const& schemaBuilder, std::shared_ptr<void> vwPredict)
    : m_state(std::make_shared<State>(schemaBuilder, std::static_pointer_cast<VWModel>(vwPredict))) {
    for (auto const& entry : m_state->m_schema) {
        m_inputs.emplace(entry.InputName, EmplaceInputEntry(entry));
        m_input_names.emplace_back(entry.InputName);
    }
    // For the sake of right now suppose we just have a single float output named "Output".
    m_outputs.emplace("Output", inference::TypeDescriptor::Create<float>());
}

Model::~Model() {}

std::unordered_map<std::string, inference::TypeDescriptor> const& Model::Inputs() const { return m_inputs; }

std::unordered_map<std::string, inference::TypeDescriptor> const& Model::Outputs() const { return m_outputs; }

std::vector<std::string> Model::GetRequirements(std::string const& outputName) const { return m_input_names; }

rt::expected<std::shared_ptr<inference::ValueUpdater>> Model::CreateValueUpdater(
    std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
    std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe,
    std::function<void()> outOfBandNotifier) const {
    // One builder per namespace, possibly shared among multiple inputs.
    std::unordered_map<std::string, std::shared_ptr<vw_slim::example_predict_builder>> builders;

    std::vector<std::unique_ptr<ValueUpdaterBase::IPeeker>> peekers;
    peekers.reserve(inputToHandle.size());
    auto vw_example = std::make_shared<::safe_example_predict>();

    // Since every input entry is listed as a requirement, every entry should in principle have a corresponding entry.
    for (auto const& entry : m_state->m_schema) {
        auto foundInputType = m_inputs.find(entry.InputName);
        // The code could continue to work if we just continue for either of these, but really if either of these
        // happens it means that whatever code is calling this is breaking the contract. All inputs are, according to
        // this object, required.
        if (foundInputType == m_inputs.end())
            return inference::make_feature_unexpected(inference::feature_errc::name_not_found);
        auto foundInputHandle = inputToHandle.find(entry.InputName);
        if (foundInputHandle == inputToHandle.end())
            return inference::make_feature_unexpected(inference::feature_errc::name_not_found);
        if (foundInputType->second != foundInputHandle->second->Type())
            return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);

        std::shared_ptr<vw_slim::example_predict_builder> builder;
        auto foundBuilder = builders.find(entry.Namespace);
        if (foundBuilder == builders.end()) {
            builder =
                std::make_shared<vw_slim::example_predict_builder>(vw_example.get(), (char*)entry.Namespace.c_str());
            builders.emplace(entry.Namespace, builder);
        } else
            builder = foundBuilder->second;
        auto peeker = ValueUpdaterBase::CreatePeeker(entry, builder, foundInputHandle->second);
        peekers.push_back(std::move(peeker));
    }

    // Map the outputs.
    auto foundOutput = outputToPipe.find("Output");
    if (foundOutput == outputToPipe.end())
        return inference::make_feature_unexpected(inference::feature_errc::name_not_found);
    if (foundOutput->second->Type() != inference::TypeDescriptor::Create<float>())
        return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);
    auto typedOutputPipe = std::static_pointer_cast<inference::DirectInputPipe<float>>(foundOutput->second);
    outOfBandNotifier();
    return std::make_shared<ValueUpdaterFloat>(m_state, vw_example, std::move(peekers), typedOutputPipe);
}

}  // namespace resonance_vw
