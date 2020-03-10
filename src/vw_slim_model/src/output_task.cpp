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
    explicit Regression(std::string outputName) : m_outputName(outputName) {
        m_outputs.emplace(outputName, inference::TypeDescriptor::Create<float>());
    }

   private:
    const std::string m_outputName;
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
        auto found = outputToPipe.find(m_outputName);
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
    explicit Recommendation(std::string outputName) : m_outputName(outputName) {
        m_outputs.emplace(outputName, inference::TypeDescriptor::Create<float>());
    }

   private:
    const std::string m_outputName;
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
        auto found = outputToPipe.find(m_outputName);
        // Just not requesting the output is not an error condition in itself.
        if (found == outputToPipe.end()) return new OutputTask::IPoker();
        if (found->second->Type() != inference::TypeDescriptor::Create<float>())
            return inference::make_feature_unexpected(inference::feature_errc::type_mismatch);
        return new Poker(vwModel, vwExample,
                         std::static_pointer_cast<inference::DirectInputPipe<float>>(found->second));
    }
};

std::shared_ptr<OutputTask> OutputTask::MakeRegression(std::string outputName) {
    return std::shared_ptr<OutputTask>(new OutputTask::Regression(outputName));
}

}  // namespace resonance_vw