// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cassert>
#include <memory>
#include <sstream>
#include <vw_common/actions.hpp>
#include <vw_common/error.hpp>
#include <vw_common/schema_builder.hpp>
#include <vw_common/schema_entry.hpp>
#include <vw_remote_model/irecommender_client.hpp>
#include <vw_remote_model/remote_model.hpp>

namespace resonance_vw {

typedef std::vector<priv::SchemaEntry> SchemaList;

template <typename T>
static std::string Convert(T const& v) {
    std::ostringstream str;
    str << v;
    return str.str().c_str();
}

template <typename T>
static std::vector<std::string> ConvertVector(std::vector<T> const& vec) {
    std::vector<std::string> stringVec(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        auto s = Convert(vec[i]);
        stringVec[i] = std::move(s);
    }

    return stringVec;
}

static std::vector<std::string> ConvertActions(std::shared_ptr<Actions> actions) {
    switch (actions->Type()) {
        case ActionType::Float:
            return ConvertVector(actions->GetFloatActions().value());
        case ActionType::Int:
            return ConvertVector(actions->GetIntActions().value());
        case ActionType::String:
            return actions->GetStringActions().value();
        case ActionType::Unknown:
        default:
            assert(false);
            return std::vector<std::string>();
            break;
    }
}

class IPeeker {
   public:
    virtual std::string PeekName() = 0;
    virtual std::string PeekValue() = 0;
};

template <typename T>
class Peeker : public IPeeker {
   public:
   protected:
    Peeker(std::string const& name, std::shared_ptr<inference::IHandle> handle)
        : m_name(name), m_handle(std::static_pointer_cast<inference::Handle<T>>(std::move(handle))) {}
    std::string m_name;
    std::shared_ptr<inference::Handle<T>> m_handle;
};

class IPoker {
   public:
    virtual void Poke(std::string const& value) = 0;
};
template <typename T>
class Poker : public IPoker {
   public:
    Poker(std::shared_ptr<inference::InputPipe> pipe) : m_pipe(std::static_pointer_cast<inference::DirectInputPipe<T>>(pipe)) {}
    virtual void Poke(std::string const& value) override;

   private:
    std::shared_ptr<inference::DirectInputPipe<T>> m_pipe;
};

template <>
void Poker<float>::Poke(std::string const& value) {
    m_pipe->Feed(std::stof(value));
}

template <>
void Poker<int>::Poke(std::string const& value) {
    m_pipe->Feed(std::stoi(value));
}

template <>
void Poker<std::string>::Poke(std::string const& value) {
    m_pipe->Feed(value);
}

static std::unique_ptr<IPeeker> CreatePeeker(priv::SchemaEntry const& entry,
                                             std::shared_ptr<inference::IHandle> handle);

class ValueUpdater : public inference::ValueUpdater {
   public:
    ValueUpdater(std::vector<std::unique_ptr<IPeeker>>&& peekers, std::vector<std::string>&& actions,
                 std::shared_ptr<IPoker> outputPoker, std::shared_ptr<IRecommenderClient> client)
        : m_peekers(std::move(peekers)), m_actions(actions), m_outputPoker(std::move(outputPoker)), m_client(client) {}

    std::error_code UpdateOutput() override;

   private:
    std::vector<std::unique_ptr<IPeeker>> m_peekers;
    std::vector<std::string> m_actions;
    std::shared_ptr<IPoker> m_outputPoker;
    std::map<std::string, std::shared_ptr<inference::InputPipe>> m_outputToPipe;
    std::shared_ptr<IRecommenderClient> m_client;
};

class IntStringPeeker : public Peeker<int> {
   public:
    IntStringPeeker(std::string const& name, std::shared_ptr<inference::IHandle> handle) : Peeker(name, handle) {}
    virtual std::string PeekValue() override { return Convert(m_handle->Value()); }
};

class FloatStringPeeker : public Peeker<float> {
   public:
    FloatStringPeeker(std::string const& name, std::shared_ptr<inference::IHandle> handle) : Peeker(name, handle) {}
    virtual std::string PeekName() override { return m_name; }
    virtual std::string PeekValue() override { return Convert(m_handle->Value()); }
};

class StringStringPeeker : public Peeker<std::string> {
   public:
    StringStringPeeker(std::string const& name, std::shared_ptr<inference::IHandle> handle) : Peeker(name, handle) {}
    virtual std::string PeekName() override { return m_name + Convert(m_handle->Value()); }
    virtual std::string PeekValue() override { return "1.f"; }
};

std::unique_ptr<IPeeker> CreatePeeker(priv::SchemaEntry const& entry, std::shared_ptr<inference::IHandle> handle) {
    switch (entry.Type) {
        case priv::SchemaType::StringString:
            return std::make_unique<StringStringPeeker>(entry.Feature, std::move(handle));
        case priv::SchemaType::FloatString:
            return std::make_unique<FloatStringPeeker>(entry.Feature, std::move(handle));
        default:
            return std::make_unique<StringStringPeeker>(entry.Feature, std::move(handle));
    }
}

std::error_code ValueUpdater::UpdateOutput() {
    // Iterate over the peekers
    std::unordered_map<std::string, std::string> features;
    for (int i = 0; i < m_peekers.size(); ++i) {
        features[m_peekers[i]->PeekName()] = m_peekers[i]->PeekValue();
    }

    auto result = m_client->GetRecommendation(features, m_actions);
    if (!result.has_value()) {
        return result.error();
    }

    m_outputPoker->Poke(result.value());
    return {};
}

rt::expected<std::shared_ptr<inference::Model>> RemoteModel::Load(SchemaBuilder const& schemaBuilder,
                                                                  std::shared_ptr<Actions> actions,
                                                                  std::shared_ptr<IRecommenderClient> client) {
    return std::shared_ptr<Model>(new RemoteModel(schemaBuilder, std::move(actions), std::move(client)));
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

RemoteModel::RemoteModel(SchemaBuilder const& schemaBuilder, std::shared_ptr<Actions> actions,
                         std::shared_ptr<IRecommenderClient> client)
    : m_schema(schemaBuilder.Schema()), m_actions(actions), m_client(client) {
    auto& schema = *std::static_pointer_cast<SchemaList>(m_schema);
    for (auto const& entry : schema) {
        m_inputs.emplace(entry.InputName, EmplaceInputEntry(entry));
        m_input_names.emplace_back(entry.InputName);
    }

    switch (actions->Type()) {
        case ActionType::Float:
            m_outputs.emplace("Output", inference::TypeDescriptor::Create<float>());
            break;
        case ActionType::Int:
            m_outputs.emplace("Output", inference::TypeDescriptor::Create<int>());
            break;
        case ActionType::String:
            m_outputs.emplace("Output", inference::TypeDescriptor::Create<std::string>());
            break;
    }
}

RemoteModel::~RemoteModel() {}

std::unordered_map<std::string, inference::TypeDescriptor> const& RemoteModel::Inputs() const { return m_inputs; }

std::unordered_map<std::string, inference::TypeDescriptor> const& RemoteModel::Outputs() const { return m_outputs; }

std::vector<std::string> RemoteModel::GetRequirements(std::string const& outputName) const { return m_input_names; }

rt::expected<std::shared_ptr<inference::ValueUpdater>> RemoteModel::CreateValueUpdater(
    std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
    std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe,
    std::function<void()> outOfBandNotifier) const {
    outOfBandNotifier();  // No out of band information, so call and ignore henceforth.

    // Validate the input schema
    auto schema = std::static_pointer_cast<SchemaList>(m_schema);
    std::vector<std::unique_ptr<IPeeker>> peekers;
    for (auto entry : *schema) {
        auto foundInput = m_inputs.find(entry.InputName);
        if (foundInput == m_inputs.end())
            return inference::make_feature_unexpected(inference::feature_errc::name_not_found);
        auto foundInputHandle = inputToHandle.find(entry.InputName);
        if (foundInputHandle == inputToHandle.end())
            return inference::make_feature_unexpected(inference::feature_errc::name_not_found);
        if (foundInput->second != foundInputHandle->second->Type()) {
            return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);
        }

        auto peeker = CreatePeeker(entry, foundInputHandle->second);
        peekers.push_back(std::move(peeker));
    }

    auto actions = ConvertActions(m_actions);
    if (actions.size() == 0) {
        return make_vw_unexpected(vw_errc::invalid_actions);
    }

    // Find the output
    auto outputPipe = outputToPipe.find("Output");
    if (outputPipe == outputToPipe.end()) {
        return inference::make_feature_unexpected(inference::feature_errc::name_not_found);
    }

    // Create the poker
    std::shared_ptr<IPoker> outputPoker;
    switch (m_actions->Type()) {
        case ActionType::Float:
            outputPoker = std::make_shared<Poker<float>>(outputPipe->second);
            break;
        case ActionType::Int:
            outputPoker = std::make_shared<Poker<int>>(outputPipe->second);
            break;
        case ActionType::String:
            outputPoker = std::make_shared<Poker<std::string>>(outputPipe->second);
            break;
        default:
            return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);
    }

    auto updater = std::make_shared<ValueUpdater>(std::move(peekers), std::move(actions), outputPoker, m_client);
    return std::static_pointer_cast<inference::ValueUpdater>(updater);
}

}  // namespace resonance_vw
