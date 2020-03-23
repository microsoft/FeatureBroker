// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <inference/direct_input_pipe.hpp>
#include <inference/feature_error.hpp>
#include <vw_common/error.hpp>
#include <vw_slim_model/output_task.hpp>

#pragma warning(push)
#pragma warning(disable : 26451)  // Arithmetic overflow warnings in hash.h.
#pragma warning(disable : 26495)  // Uninitialized members and variables seemingly everywhere.
#pragma warning(disable : 26812)  // Enums used throughout instad of enum class.
#include "array_parameters.h"
#include "example_predict_builder.h"
#include "vw_slim_predict.h"
#include "vw_slim_return_codes.h"
#pragma warning(pop)

typedef vw_slim::vw_predict<::sparse_parameters> VWModel;
typedef ::safe_example_predict VWExample;

namespace {
tl::expected<inference::TypeDescriptor, std::error_code> ActionTypeToType(
    const resonance_vw::ActionType atype) noexcept {
    switch (atype) {
        case resonance_vw::ActionType::Float:
            return inference::TypeDescriptor::Create<inference::Tensor<float>>();
        case resonance_vw::ActionType::Int:
            return inference::TypeDescriptor::Create<inference::Tensor<int>>();
        case resonance_vw::ActionType::String:
            return inference::TypeDescriptor::Create<inference::Tensor<std::string>>();
    }
    return tl::make_unexpected(inference::make_feature_error(inference::feature_errc::type_unsupported));
}

size_t CountOfActions(std::shared_ptr<resonance_vw::Actions> const& actions) noexcept {
    switch (actions->Type()) {
        case resonance_vw::ActionType::Int:
            return actions->GetIntActions().value().size();
        case resonance_vw::ActionType::String:
            return actions->GetStringActions().value().size();
    }
    return 0;
}
}  // namespace

namespace resonance_vw {

class OutputTask::IPokerImpl : public IPoker {
   protected:
    const std::shared_ptr<VWModel> m_model;
    const std::shared_ptr<VWExample> m_example;

   public:
    IPokerImpl(std::shared_ptr<void> vwModel, std::shared_ptr<void> vwExample);
    ~IPokerImpl() = default;
};

OutputTask::IPokerImpl::IPokerImpl(std::shared_ptr<void> vwModel, std::shared_ptr<void> vwExample)
    : m_model(std::static_pointer_cast<VWModel>(vwModel)), m_example(std::static_pointer_cast<VWExample>(vwExample)) {}

OutputTask::IPoker::IPoker() {}
OutputTask::IPoker::~IPoker() {}
std::error_code OutputTask::IPoker::Poke() { return {}; }

OutputTask::OutputTask() {}
OutputTask::~OutputTask() {}

class OutputTask::Regression final : public OutputTask {
   public:
    Regression() { m_outputs.emplace("Output", inference::TypeDescriptor::Create<float>()); }

   private:
    std::unordered_map<std::string, inference::TypeDescriptor> m_outputs;

    class Poker final : public IPokerImpl {
       public:
        Poker(std::shared_ptr<void> vwModel, std::shared_ptr<void> vwExample,
              std::shared_ptr<inference::DirectInputPipe<float>> pipe)
            : IPokerImpl(vwModel, vwExample), m_pipe(pipe) {}
        ~Poker() = default;

        std::error_code Poke() override {
            float result = 0;
            int err = m_model->predict(*m_example, result);
            if (err) return make_vw_error(vw_errc::predict_failure);
            m_pipe->Feed(result);
            return {};
        }

       private:
        const std::shared_ptr<inference::DirectInputPipe<float>> m_pipe;
    };

    std::unordered_map<std::string, inference::TypeDescriptor> const& Outputs() const { return m_outputs; }

    tl::expected<OutputTask::IPoker*, std::error_code> CreatePoker(
        std::shared_ptr<void> vwModel, std::shared_ptr<void> vwExample,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe) const override {
        auto found = outputToPipe.find("Output");
        // Just not requesting the output is not an error condition in itself.
        if (found == outputToPipe.end()) return new OutputTask::IPoker();
        if (found->second->Type() != inference::TypeDescriptor::Create<float>())
            return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);
        return new Poker(vwModel, vwExample,
                         std::static_pointer_cast<inference::DirectInputPipe<float>>(found->second));
    }
};

class OutputTask::Recommendation final : public OutputTask {
   public:
    Recommendation(std::shared_ptr<Actions> actions, inference::TypeDescriptor actionsType,
                   std::string const& experimentId, std::string const& ns)
        : m_state(std::make_shared<State>(experimentId, ns, actions)) {
        m_outputs.emplace("Actions", actionsType);
        m_outputs.emplace("Indices", inference::TypeDescriptor::Create<inference::Tensor<int>>());
        m_outputs.emplace("Probabilities", inference::TypeDescriptor::Create<inference::Tensor<float>>());
    }

   private:
    class State final {
       public:
        State(std::string const& eid, std::string const& cns, std::shared_ptr<Actions> actions)
            : m_eid(eid), m_actions(std::move(actions)) {
            m_actionEx.resize(::CountOfActions(m_actions));
            char* ns = (char*)cns.c_str();  // Ewww. The signatures to VW Slim could use some qualifiers.
            for (int i = 0; i < m_actionEx.size(); ++i) {
                vw_slim::example_predict_builder(&m_actionEx[i], ns).push_feature(i, 1.f);
            }
        }

        template <typename T>
        std::vector<T> ActionVector() = delete;

        const std::string m_eid;
        const std::shared_ptr<Actions> m_actions;
        std::vector<VWExample> m_actionEx;
    };

    std::shared_ptr<State> m_state;
    std::unordered_map<std::string, inference::TypeDescriptor> m_outputs;

    template <typename T>
    class Poker : public IPokerImpl {
       public:
        Poker(std::shared_ptr<void> vwModel, std::shared_ptr<void> vwExample, std::shared_ptr<State> state,
              std::shared_ptr<inference::InputPipe> apipe, std::shared_ptr<inference::InputPipe> ipipe,
              std::shared_ptr<inference::InputPipe> ppipe)
            : IPokerImpl(vwModel, vwExample),
              m_state(state),
              m_actionsPipe(std::static_pointer_cast<inference::DirectInputPipe<inference::Tensor<T>>>(apipe)),
              m_indicesPipe(std::static_pointer_cast<inference::DirectInputPipe<inference::Tensor<int>>>(ipipe)),
              m_probPipe(std::static_pointer_cast<inference::DirectInputPipe<inference::Tensor<float>>>(ppipe)),
              m_dims({state->m_actionEx.size()}) {}
        ~Poker() = default;

        std::error_code Poke() override {
            int err = m_model->predict(m_state->m_eid.c_str(), *m_example, m_state->m_actionEx.data(),
                                       m_state->m_actionEx.size(), m_pdfs, m_rankings);
            if (err) return make_vw_error(vw_errc::predict_failure);

            size_t size = m_state->m_actionEx.size();

            // C++17 can handle this a bit more gracefully.
            std::shared_ptr<int> indices(new int[size], [](int* a) { delete[] a; });
            std::shared_ptr<float> pdfs(new float[size], [](float* a) { delete[] a; });
            std::shared_ptr<T> actions(new T[size], [](T* a) { delete[] a; });
            // This copying out is inefficient. The action holder may need some adjustment.
            auto avalues = m_state->ActionVector<T>();

            for (size_t i = 0; i < size; ++i) {
                indices.get()[i] = m_rankings[i];
                actions.get()[i] = avalues[indices.get()[i]];
                pdfs.get()[i] = m_pdfs[i];
            }

            if (m_actionsPipe) m_actionsPipe->Feed(inference::Tensor<T>(actions, m_dims));
            if (m_indicesPipe) m_indicesPipe->Feed(inference::Tensor<int>(indices, m_dims));
            if (m_probPipe) m_probPipe->Feed(inference::Tensor<float>(pdfs, m_dims));

            return {};
        }

       private:
        const std::shared_ptr<State> m_state;
        const std::shared_ptr<inference::DirectInputPipe<inference::Tensor<T>>> m_actionsPipe;
        const std::shared_ptr<inference::DirectInputPipe<inference::Tensor<int>>> m_indicesPipe;
        const std::shared_ptr<inference::DirectInputPipe<inference::Tensor<float>>> m_probPipe;
        const std::vector<size_t> m_dims;

        std::vector<int> m_rankings;
        std::vector<float> m_pdfs;
    };

    std::unordered_map<std::string, inference::TypeDescriptor> const& Outputs() const { return m_outputs; }

    tl::expected<OutputTask::IPoker*, std::error_code> CreatePoker(
        std::shared_ptr<void> vwModel, std::shared_ptr<void> vwExample,
        std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe) const override {
        auto actionTypeExpected = ::ActionTypeToType(m_state->m_actions->Type());
        if (!actionTypeExpected) return tl::make_unexpected(actionTypeExpected.error());

        auto found = outputToPipe.find("Actions");
        std::shared_ptr<inference::InputPipe> actionPipe = nullptr;
        if (found != outputToPipe.end()) {
            actionPipe = found->second;
            if (actionPipe->Type() != actionTypeExpected.value())
                return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);
        }

        found = outputToPipe.find("Indices");
        auto indicesPipe = found == outputToPipe.end() ? nullptr : found->second;
        if (indicesPipe != nullptr &&
            indicesPipe->Type() != inference::TypeDescriptor::Create<inference::Tensor<int>>()) {
            return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);
        }

        found = outputToPipe.find("Probabilities");
        auto probPipe = found == outputToPipe.end() ? nullptr : found->second;
        if (probPipe != nullptr && probPipe->Type() != inference::TypeDescriptor::Create<inference::Tensor<float>>()) {
            return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);
        }

        // Just not requesting the output is not an error condition in itself.
        if (actionPipe == nullptr && indicesPipe == nullptr && probPipe == nullptr) return new OutputTask::IPoker();
        switch (m_state->m_actions->Type()) {
            case resonance_vw::ActionType::Float:
                return new Poker<float>(vwModel, vwExample, m_state, actionPipe, indicesPipe, probPipe);
            case resonance_vw::ActionType::Int:
                return new Poker<int>(vwModel, vwExample, m_state, actionPipe, indicesPipe, probPipe);
            case resonance_vw::ActionType::String:
                return new Poker<std::string>(vwModel, vwExample, m_state, actionPipe, indicesPipe, probPipe);
        }
        return inference::make_feature_unexpected(inference::feature_errc::type_unsupported);
    }
};

std::shared_ptr<OutputTask> OutputTask::MakeRegression() {
    return std::shared_ptr<OutputTask>(new OutputTask::Regression());
}

tl::expected<std::shared_ptr<OutputTask>, std::error_code> OutputTask::MakeRecommendation(
    std::shared_ptr<Actions> actions, std::string const& experimentId, std::string const& classNamesapce) {
    auto actionsTypeExpected = ::ActionTypeToType(actions->Type());
    if (!actionsTypeExpected) return tl::make_unexpected(actionsTypeExpected.error());

    auto task = std::make_shared<OutputTask::Recommendation>(actions, actionsTypeExpected.value(), experimentId,
                                                             classNamesapce);
    return task;
}

template <>
std::vector<float> OutputTask::Recommendation::State::ActionVector<float>() {
    return m_actions->GetFloatActions().value();
}

template <>
std::vector<int> OutputTask::Recommendation::State::ActionVector<int>() {
    return m_actions->GetIntActions().value();
}

template <>
std::vector<std::string> OutputTask::Recommendation::State::ActionVector<std::string>() {
    return m_actions->GetStringActions().value();
}

}  // namespace resonance_vw