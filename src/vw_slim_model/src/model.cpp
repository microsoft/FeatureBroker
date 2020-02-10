#ifndef _WIN32
// There appear to be some includes missing from memory.h that let it compile.
// Also, VW uses throw within memory.h despite VW_NOEXCEPT being set for some
// reason, and this only fails on Linux because the requisite defines needed to
// trigger the compilation of that code are not defined in Windows.
#include <unistd.h>  // So memory.h has access to _SC_PAGE_SIZE.
#define THROW        // So VW's memory.h doesn't fail when THROW is encountered. This seems subideal.
#endif

#include <vw_slim_model/model.hpp>
#include <vw_slim_model/vw_error.hpp>

#include "schema_entry.hpp"

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

namespace vw_slim_model {

typedef std::vector<priv::SchemaEntry> SchemaList;
typedef vw_slim::vw_predict<::sparse_parameters> VWModel;

class ValueUpdater : inference::ValueUpdater {
   public:
    ValueUpdater(std::shared_ptr<void> vwPredict);
    std::error_code UpdateOutput() override;

    class IPeeker {
       public:
        virtual void Peek() = 0;
        virtual ~IPeeker() = default;

       protected:
        explicit IPeeker(std::shared_ptr<vw_slim::example_predict_builder> ex);
        const std::shared_ptr<vw_slim::example_predict_builder> m_builder;
    };

    std::unique_ptr<IPeeker> CreatePeeker(priv::SchemaEntry const& entry,
                                          std::shared_ptr<vw_slim::example_predict_builder> ex,
                                          std::shared_ptr<inference::IHandle> handle);

    template <typename T>
    class Peeker : public IPeeker {
       public:
        virtual ~Peeker() = default;

       protected:
        const std::shared_ptr<vw_slim::example_predict_builder> m_builder;
        const std::shared_ptr<inference::Handle<T>> m_handle;
        Peeker(std::shared_ptr<vw_slim::example_predict_builder> ex, std::shared_ptr<inference::IHandle> handle);
    };

   private:
    const std::shared_ptr<VWModel> m_vw_predict;
    ::safe_example_predict m_vw_example;
    std::vector<IPeeker> m_peekers;
};

ValueUpdater::IPeeker::IPeeker(std::shared_ptr<vw_slim::example_predict_builder> ex) : m_builder(ex) {}

template <typename T>
ValueUpdater::Peeker<T>::Peeker(std::shared_ptr<vw_slim::example_predict_builder> ex,
                                std::shared_ptr<inference::IHandle> handle)
    : IPeeker(ex), m_handle(std::static_pointer_cast<inference::Handle<T>>(handle)) {}

class FloatIndexPeeker final : public ValueUpdater::Peeker<float> {
   public:
    void Peek() { m_builder->push_feature(m_index, m_handle->Value()); }
    FloatIndexPeeker(priv::SchemaEntry const& entry, std::shared_ptr<vw_slim::example_predict_builder> ex,
                     std::shared_ptr<inference::IHandle> handle)
        : ValueUpdater::Peeker<float>(ex, handle), m_index(entry.Index) {}

   private:
    const decltype(priv::SchemaEntry::Index) m_index;
};

class FloatStringPeeker final : public ValueUpdater::Peeker<float> {
   public:
    void Peek() {
        // Ewwww. The VW code doesn't actually modify this string at all, it just Murmur hashes it, so in principle they
        // could have put a const qualifier on the method argument, but they didn't, so, here we are. Perhaps we could
        // change their code to add const and other modifiers were possible, and remove this cast in future versions.
        m_builder->push_feature_string(const_cast<char*>(m_index.c_str()), m_handle->Value());
    }
    FloatStringPeeker(priv::SchemaEntry const& entry, std::shared_ptr<vw_slim::example_predict_builder> ex,
                      std::shared_ptr<inference::IHandle> handle)
        : ValueUpdater::Peeker<float>(ex, handle), m_index(entry.Feature) {}

   private:
    const decltype(priv::SchemaEntry::Feature) m_index;
};

class FloatsIndexPeeker final : public ValueUpdater::Peeker<inference::Tensor<float>> {
   public:
    void Peek() {
        auto val = m_handle->Value();
        auto pdata = val.Data();
        size_t total = 1;
        for (const auto d : val.Dimensions()) total *= d;
        for (size_t i = 0; i < total; ++i) {
            // This test will also result in NaN being skipped. Is that desirable/undesirable?
            if (pdata[i] != 0) m_builder->push_feature(i + m_index, pdata[i]);
        }
    }
    FloatsIndexPeeker(priv::SchemaEntry const& entry, std::shared_ptr<vw_slim::example_predict_builder> ex,
                      std::shared_ptr<inference::IHandle> handle)
        : ValueUpdater::Peeker<inference::Tensor<float>>(ex, handle), m_index(entry.Index) {}

   private:
    const decltype(priv::SchemaEntry::Index) m_index;
};

class StringStringPeeker final : public ValueUpdater::Peeker<std::string> {
   public:
    void Peek() {
        // Again eww.
        m_builder->push_feature_string(const_cast<char*>(m_handle->Value().c_str()), 1.0f);
    }
    StringStringPeeker(std::shared_ptr<vw_slim::example_predict_builder> ex, std::shared_ptr<inference::IHandle> handle)
        : ValueUpdater::Peeker<std::string>(ex, handle) {}
};

class StringsStringPeeker final : public ValueUpdater::Peeker<inference::Tensor<std::string>> {
   public:
    void Peek() {
        auto val = m_handle->Value();
        auto pdata = val.Data();
        size_t total = 1;
        for (const auto d : val.Dimensions()) total *= d;
        for (size_t i = 0; i < total; ++i) {
            // This test will also result in NaN being skipped. Is that desirable/undesirable?
            m_builder->push_feature_string(const_cast<char*>(pdata[i].c_str()), 1.0f);
        }
    }
    StringsStringPeeker(std::shared_ptr<vw_slim::example_predict_builder> ex,
                        std::shared_ptr<inference::IHandle> handle)
        : ValueUpdater::Peeker<inference::Tensor<std::string>>(ex, handle) {}
};

std::unique_ptr<ValueUpdater::IPeeker> ValueUpdater::CreatePeeker(priv::SchemaEntry const& entry,
                                                                  std::shared_ptr<vw_slim::example_predict_builder> ex,
                                                                  std::shared_ptr<inference::IHandle> handle) {
    switch (entry.Type) {
        case priv::SchemaType::FloatIndex:
            return std::make_unique<FloatIndexPeeker>(entry, ex, handle);
        case priv::SchemaType::FloatString:
            return std::make_unique<FloatStringPeeker>(entry, ex, handle);
        case priv::SchemaType::FloatsIndex:
            return std::make_unique<FloatsIndexPeeker>(entry, ex, handle);
        case priv::SchemaType::StringString:
            return std::make_unique<StringStringPeeker>(ex, handle);
        default:
            assert(entry.Type == priv::SchemaType::StringsString);
            return std::make_unique<StringsStringPeeker>(ex, handle);
    }
}

// ValueUpdater::Peekers ----------------------------------------

ValueUpdater::ValueUpdater(std::shared_ptr<void> vwPredict)
    : m_vw_predict(std::static_pointer_cast<VWModel>(vwPredict)) {}

// ValueUpdater::Peekers ----------------------------------------

std::error_code ValueUpdater::UpdateOutput() { return {}; }

rt::expected<std::shared_ptr<Model>> Model::Load(SchemaBuilder const& schemaBuilder,
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
    : m_schema(std::make_shared<SchemaList>(*std::static_pointer_cast<SchemaList>(schemaBuilder.m_schema))),
      m_vw_predict(std::move(vwPredict)) {
    auto& schema = *std::static_pointer_cast<SchemaList>(m_schema);
    for (auto const& entry : schema) {
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
    return {};
}

}  // namespace vw_slim_model
