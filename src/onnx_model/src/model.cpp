// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <onnxruntime_c_api.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <inference/direct_input_pipe.hpp>
#include <inference/handle.hpp>
#include <inference/input_pipe.hpp>
#include <inference/tensor.hpp>
#include <mutex>
#include <onnx_model/model.hpp>
#include <onnx_model/onnx_error.hpp>

namespace {

#ifdef WIN32
std::wstring Path(std::string const& name) {
    std::wstring wpath(name.size(), L' ');
    std::mbstowcs(&wpath[0], name.c_str(), name.size());
    return wpath;
}
#else
std::string Path(std::string const& name) { return name; }
#endif

/**
 * @brief Helper function for creating ONNX sessions.
 *
 * @param sessionCreator A function that given an environment sets the session parameter to be a created session, and
 * returns the ONNX runtime status.
 * @param env The ONNX environment, used to create the sessions.
 * @param session The session that was created, or null if some problem occurred.
 * @return inference::onnx_errc The error code.
 */
onnx_model::onnx_errc create_session(
    std::function<OrtStatus*(OrtEnv*, OrtSessionOptions*, OrtSession*& session)> sessionCreator, OrtEnv*& env,
    OrtSession*& session) {
    auto status = OrtCreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL, "onnx_model", &env);
    if (status != nullptr) {
        OrtReleaseStatus(status);
        return onnx_model::onnx_errc::internal_library_error;
    }
    // Create the session
    OrtSessionOptions* sessionOptions;
    if (status = OrtCreateSessionOptions(&sessionOptions)) {
        OrtReleaseStatus(status);
        OrtReleaseEnv(env);
        env = nullptr;
        return onnx_model::onnx_errc::internal_library_error;
    }
    // Create the session.
    status = sessionCreator(env, sessionOptions, session);
    OrtReleaseSessionOptions(sessionOptions);
    if (status != nullptr) {
        OrtReleaseStatus(status);
        OrtReleaseEnv(env);
        env = nullptr;
        return onnx_model::onnx_errc::model_load_error;
    }
    return onnx_model::onnx_errc();
}

/**
 * @brief A shorthand utility function for creating a type descriptor of a tensor with a given element type.
 *
 * @tparam T The element type for the type.
 * @return inference::TypeDescriptor The tensor type descriptor with this element type.
 */
template <typename T>
inline inference::TypeDescriptor MakeTensorType() {
    return inference::TypeDescriptor::Create<inference::Tensor<T>>();
}

#pragma warning(push)
#pragma warning(disable : 26812)  // Enums used throughout instad of enum class.
rt::expected<inference::TypeDescriptor> MakeType(const ONNXTensorElementDataType elementType) {
#pragma warning(pop)
    switch (elementType) {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return MakeTensorType<float_t>();
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return MakeTensorType<double_t>();
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return MakeTensorType<int32_t>();
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return MakeTensorType<int64_t>();
        default:
            return onnx_model::make_onnx_unexpected(onnx_model::onnx_errc::unsupported_type);
    }
}

/**
 * @brief Maps an element type back to a ONNX runtime element type.
 *
 * This method is deleted for those types not yet supported by this wrapping.
 *
 * @tparam T The element type.
 * @return ONNXTensorElementDataType The ONNX runtime element type.
 */
template <typename T>
ONNXTensorElementDataType MapToOrtType() = delete;

template <>
ONNXTensorElementDataType MapToOrtType<float_t>() {
    return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
template <>
ONNXTensorElementDataType MapToOrtType<double_t>() {
    return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}
template <>
ONNXTensorElementDataType MapToOrtType<int32_t>() {
    return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}
template <>
ONNXTensorElementDataType MapToOrtType<int64_t>() {
    return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

/**
 * @brief A helper function that, given a session, extracts types for either inputs or outputs.
 *
 * @tparam TCountF The type of the count function.
 * @tparam TNameF The type of the name function.
 * @tparam TTypeF The type of the type function.
 *
 * @param map The map that is filled in with type information.
 * @param alloc The runtime allocation object, used for extracting the names.
 * @param sess The session over which we are extracting names and types.
 * @param countf Intended to be one of OrtSessionGetInputCount or OrtSessionGetOutputCount.
 * @param namef Intended to be one of OrtSessionGetInputName or OrtSessionGetOutputName.
 * @param typef Intended to be one of OrtSessionGetInputTypeInfo or OrtSessionGetOutputTypeInfo.
 * @return inference::onnx_errc In the event that there is a failure, this will contain a non-zero error code.
 */
template <typename TCountF, typename TNameF, typename TTypeF>
onnx_model::onnx_errc fetch_names_types(std::unordered_map<std::string, inference::TypeDescriptor>& map,
                                        OrtAllocator* alloc, OrtSession* sess, TCountF& countf, TNameF& namef,
                                        TTypeF& typef) {
    const auto defaultError = onnx_model::onnx_errc::internal_library_error;
    size_t count = {};
    auto status = countf(sess, &count);
    if (status) {
        OrtReleaseStatus(status);
        return defaultError;
    }

    auto name_deleter = [alloc](char* c) { alloc->Free(alloc, c); };
    auto typeInfo_deleter = [](auto c) { OrtReleaseTypeInfo(c); };
    auto typeShape_deleter = [](auto c) { OrtReleaseTensorTypeAndShapeInfo(c); };

    for (size_t i = 0; i < count; ++i) {
        char* name_raw = nullptr;
        if (status = namef(sess, i, alloc, &name_raw)) {
            OrtReleaseStatus(status);
            return defaultError;
        }
        // Note the use of the custom deleter. This is a pattern we use in interacting with the ONNX library -- since
        // any call could in principle fail and there are a lot of objects that are created as a side effect of
        // interrogating the session, we put them in these scoped pointers.

        // Given the limited lifetime and the fact that they are not used beyond this loop unique pointers would have
        // been preferable, except that there appear to be some additional requirements on deleters for unique pointers
        // that make them incompatible with the capturing lambdas above, somehow.
        std::shared_ptr<char> name(name_raw, name_deleter);

        OrtTypeInfo* typeInfo_raw = nullptr;
        if (status = typef(sess, i, &typeInfo_raw)) {
            OrtReleaseStatus(status);
            return defaultError;
        }
        std::shared_ptr<OrtTypeInfo> typeInfo(typeInfo_raw, typeInfo_deleter);

        ONNXType type;
        if (status = OrtOnnxTypeFromTypeInfo(typeInfo.get(), &type)) {
            OrtReleaseStatus(status);
            return defaultError;
        }
        // We do not support any other type but tensors. (E.g., no maps, etc.)
        if (type != ONNXType::ONNX_TYPE_TENSOR) return onnx_model::onnx_errc::unsupported_type;

        const OrtTensorTypeAndShapeInfo* typeShape = nullptr;
        if (status = OrtCastTypeInfoToTensorInfo(typeInfo.get(), &typeShape)) {
            OrtReleaseStatus(status);
            return defaultError;
        }
        // Note that we are not wrapping the type-shape in a scoped pointer since according to the documentation, that
        // object is not allocated.

        ONNXTensorElementDataType elementType;
        if (status = OrtGetTensorElementType(typeShape, &elementType)) {
            OrtReleaseStatus(status);
            return defaultError;
        }

        // OK. We have now extracted the type. Enter it into the map.
        auto typeExpected = MakeType(elementType);
        if (!typeExpected) return onnx_model::onnx_errc::unsupported_type;
        map.emplace(name.get(), typeExpected.value());
    }
    return onnx_model::onnx_errc();
}
}  // namespace

namespace onnx_model {

/**
 * The state class hides away all ONNX runtime library specific structures, so that no ONNX runtime library structures
 * are exposed in the ONNXModel class declaration and header.
 */
class Model::State final {
   public:
    const std::shared_ptr<OrtEnv> m_env;
    const std::shared_ptr<OrtSession> m_session;
    std::shared_ptr<OrtAllocator> m_alloc;
    std::shared_ptr<OrtAllocatorInfo> m_allocInfo;
    // Unlike, say, TensorFlow, ONNX runtime appears to make no distinction between a graph and a session, and unless
    // the session is allocating fresh memory for all intermediate tensors on every run (which would be grossly
    // inefficient), it is not reasonable to suppose that a session is thread safe. We might inquire after this.
    std::mutex m_model_lock;

    State(OrtEnv* env, OrtSession* session)
        : m_env(std::shared_ptr<OrtEnv>(env, [this](OrtEnv* e) { OrtReleaseEnv(e); })),
          m_session(std::shared_ptr<OrtSession>(session, [this](OrtSession* s) { OrtReleaseSession(s); })) {
        OrtAllocator* alloc = nullptr;
        auto status = OrtCreateDefaultAllocator(&alloc);
        if (status != nullptr) alloc = nullptr;
        m_alloc = std::shared_ptr<OrtAllocator>(alloc, [this](OrtAllocator* a) { OrtReleaseAllocator(a); });

        const OrtAllocatorType allocator_type = OrtArenaAllocator;
        const OrtMemType mem_type = OrtMemTypeDefault;
        OrtAllocatorInfo* allocInfo;
        // Should this not be working through OrtAllocatorGetInfo instead?
        if (status = OrtCreateCpuAllocatorInfo(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault,
                                               &allocInfo)) {
            m_allocInfo = nullptr;
        } else {
            m_allocInfo = std::shared_ptr<OrtAllocatorInfo>(
                allocInfo, [this](OrtAllocatorInfo* ai) { OrtReleaseAllocatorInfo(ai); });
        }
    }
};

/**
 * @brief Construct a new ONNX Model.
 *
 * This is a private constructor called from one of the static load methods.
 *
 * @param env The ONNX environment.
 * @param session The session that this ONNX model is wrapping.
 * @param errc A reference error-code that is set to something non-zero in case anything goes wrong during construction.
 */
Model::Model(void* env, void* session, onnx_errc& errc)
    : m_state(std::shared_ptr<State>(new State(static_cast<OrtEnv*>(env), static_cast<OrtSession*>(session)))) {
    // Initialize the map of the inputs.
    if (!m_state->m_allocInfo) {
        errc = onnx_errc::internal_library_error;
        return;
    }
    size_t count = 0;
    auto sess = m_state->m_session.get();
    auto err = ::fetch_names_types(m_inputs, m_state->m_alloc.get(), m_state->m_session.get(), OrtSessionGetInputCount,
                                   OrtSessionGetInputName, OrtSessionGetInputTypeInfo);
    if (static_cast<bool>(err)) {
        errc = err;
        return;
    }
    err = ::fetch_names_types(m_outputs, m_state->m_alloc.get(), m_state->m_session.get(), OrtSessionGetOutputCount,
                              OrtSessionGetOutputName, OrtSessionGetOutputTypeInfo);
    if (static_cast<bool>(err)) {
        errc = err;
        return;
    }
    for (const auto& pair : m_inputs) m_deps.push_back(pair.first);
}

rt::expected<std::shared_ptr<Model>> Model::Load(std::string const& path) noexcept {
    OrtEnv* env;
    OrtSession* session;
    onnx_errc errc = ::create_session(
        [&path](OrtEnv* dEnv, OrtSessionOptions* dOpt, OrtSession*& dSess) {
            auto load_path = Path(path);
            return OrtCreateSession(dEnv, load_path.c_str(), dOpt, &dSess);
        },
        env, session);
    if (static_cast<int>(errc) != 0) return onnx_model::make_onnx_unexpected(errc);
    auto model = std::shared_ptr<Model>(new Model(env, session, errc));
    if (static_cast<int>(errc) != 0) return onnx_model::make_onnx_unexpected(errc);
    return model;
}

rt::expected<std::shared_ptr<Model>> Model::LoadFromBuffer(const void* modelData, const size_t modelSize) noexcept {
    OrtEnv* env;
    OrtSession* session;
    onnx_errc errc = ::create_session(
        [&modelData, &modelSize](OrtEnv* dEnv, OrtSessionOptions* dOpt, OrtSession*& dSess) {
            return OrtCreateSessionFromArray(dEnv, modelData, modelSize, dOpt, &dSess);
        },
        env, session);
    if (static_cast<int>(errc) != 0) return onnx_model::make_onnx_unexpected(errc);
    auto model = std::shared_ptr<Model>(new Model(env, session, errc));
    if (static_cast<int>(errc) != 0) return onnx_model::make_onnx_unexpected(errc);
    return model;
}

Model::~Model() = default;

std::unordered_map<std::string, inference::TypeDescriptor> const& Model::Inputs() const { return m_inputs; }
std::unordered_map<std::string, inference::TypeDescriptor> const& Model::Outputs() const { return m_outputs; }

std::vector<std::string> Model::GetRequirements(std::string const& outputName) const {
    // For the sake of this API. The ONNX C graph API does not appear to have any mechanism to traverse
    // the ONNX graph like the Python API does, and so be more specific with dependencies.
    return m_deps;
}

/**
 * @brief An abstract interface for mapping handles into input tensors.
 */
class IPeeker {
   public:
    /**
     * @brief Maps the handle's pointer into a wrapping ONNX tensor value, in preparation for OrtRun.
     *
     * @param value An ONNX runtime tensor value that is a simple wrapper of the handle's pointer.
     * @param ai The allocator to use when allocating the object.
     * @return onnx_errc If non-zero, an error that occurred.
     */
    virtual onnx_errc Peek(OrtValue*& value, const OrtAllocatorInfo* ai) = 0;
    virtual ~IPeeker() = default;
    static std::unique_ptr<IPeeker> CreatePeeker(std::shared_ptr<inference::IHandle> handle) noexcept;

   protected:
    onnx_errc PeekCore(OrtValue*& value, const OrtAllocatorInfo* ai, ONNXTensorElementDataType const ortType,
                       const size_t sizeOfT, std::vector<size_t> const& dims, void* data);
};

onnx_errc IPeeker::PeekCore(OrtValue*& value, const OrtAllocatorInfo* ai, ONNXTensorElementDataType const ortType,
                            const size_t sizeOfT, std::vector<size_t> const& dims, void* data) {
    if (value != nullptr) OrtReleaseValue(value);
    size_t total = sizeOfT;
    for (const size_t& dim : dims) total *= dim;

    OrtStatus* status;
    if (sizeof(size_t) == sizeof(int64_t)) {
        // If the same size, we suppose here that a reinterpret cast is sufficient.
        status = OrtCreateTensorWithDataAsOrtValue(ai, data, total, reinterpret_cast<const int64_t*>(dims.data()),
                                                   dims.size(), ortType, &value);
    } else {
        // If they're not the same size we explicitly allocate and cast to a new block of memory.
        std::vector<int64_t> shape;
        shape.reserve(dims.size());
        for (const size_t& dim : dims) shape.push_back(static_cast<const int64_t>(dim));
        status = OrtCreateTensorWithDataAsOrtValue(ai, data, total, shape.data(), dims.size(), ortType, &value);
    }

    if (status != nullptr) {
        OrtReleaseStatus(status);
        return onnx_errc::internal_library_error;
    }
    return onnx_errc();
}

template <typename T>
class Peeker final : public IPeeker {
   private:
    std::shared_ptr<inference::Handle<inference::Tensor<T>>> m_handle;

   public:
    Peeker(std::shared_ptr<inference::IHandle> handle)
        : m_handle(std::static_pointer_cast<inference::Handle<inference::Tensor<T>>>(std::move(handle))) {}
    ~Peeker() = default;

    onnx_errc Peek(OrtValue*& value, const OrtAllocatorInfo* ai) override;
};

template <typename T>
onnx_errc Peeker<T>::Peek(OrtValue*& value, const OrtAllocatorInfo* ai) {
    // If it has not changed from the last call, then it must be the same as when we last wrapped it, so there is no
    // need to change anything.
    if (!m_handle->Changed()) return onnx_errc();
    inference::Tensor<T> tensor = m_handle->Value();
    return PeekCore(value, ai, ::MapToOrtType<T>(), sizeof(T), tensor.Dimensions(), tensor.Data());
}

/**
 * @brief An abstract interface for mapping output tensors to pipes. Contrast with its counterpart, the peeker.
 */
class IPoker {
   protected:
    virtual void FeedCore(std::shared_ptr<void>& data, std::vector<size_t> const& dims) = 0;
    virtual ONNXTensorElementDataType ExpectedElementType() const = 0;

   public:
    virtual ~IPoker() = default;
    /**
     * @brief Takes the output ONNX tensor value and creates an output tensor value, after a successful call to OrtRun.
     *
     * @param value The value out of which we fetched the value.
     * @param ai
     * @return onnx_errc
     */
    onnx_errc Poke(OrtValue* value, const OrtAllocatorInfo* ai);
    static std::unique_ptr<IPoker> CreatePoker(std::shared_ptr<inference::InputPipe> pipe) noexcept;
};

onnx_errc IPoker::Poke(OrtValue* value, const OrtAllocatorInfo* ai) {
    int isTensor;
    OrtStatus* status;
    // Note that we do a series of type checks here. In principle this could
    // only happen if the value produced differs from the value advertised
    // by the session prior to running, but we check anyway.
    if (status = OrtIsTensor(value, &isTensor)) {
        OrtReleaseStatus(status);
        return onnx_errc::internal_library_error;
    }
    if (isTensor == 0) return onnx_errc::type_mismatch;

    OrtTypeInfo* typeInfoRaw;
    if (status = OrtGetTypeInfo(value, &typeInfoRaw)) {
        OrtReleaseStatus(status);
        return onnx_errc::internal_library_error;
    }
    auto typeInfo = std::shared_ptr<OrtTypeInfo>(typeInfoRaw, [](OrtTypeInfo* ti) { OrtReleaseTypeInfo(ti); });
    const OrtTensorTypeAndShapeInfo* typeShape;
    if (status = OrtCastTypeInfoToTensorInfo(typeInfo.get(), &typeShape)) {
        OrtReleaseStatus(status);
        return onnx_errc::internal_library_error;
    }
    ONNXTensorElementDataType elementType;
    if (status = OrtGetTensorElementType(typeShape, &elementType)) {
        OrtReleaseStatus(status);
        return onnx_errc::internal_library_error;
    }
    if (ExpectedElementType() != elementType) return onnx_errc::type_mismatch;
    // The type checks track. Get the dimensions and data, and wrap them into a tensor.
    std::vector<size_t> dims;
    size_t dimsLen;
    if (status = OrtGetDimensionsCount(typeShape, &dimsLen)) {
        OrtReleaseStatus(status);
        return onnx_errc::internal_library_error;
    }
    std::vector<int64_t> dimsTemp;
    dimsTemp.resize(dimsLen);
    if (status = OrtGetDimensions(typeShape, dimsTemp.data(), dimsLen)) {
        OrtReleaseStatus(status);
        return onnx_errc::internal_library_error;
    }
    void* data;
    if (status = OrtGetTensorMutableData(value, &data)) {
        OrtReleaseStatus(status);
        return onnx_errc::internal_library_error;
    }
    dims.reserve(dimsLen);
    for (const auto& d : dimsTemp) dims.push_back(static_cast<size_t>(d));
    // Note that by having a shared pointer with a custom deleter, we avoid doing any copy.
    auto deleter = [value](void* p) { OrtReleaseValue(value); };
    std::shared_ptr<void> dataShared(data, deleter);
    FeedCore(dataShared, dims);
    return onnx_errc();
}

template <typename T>
class Poker final : public IPoker {
   private:
    std::shared_ptr<inference::DirectInputPipe<inference::Tensor<T>>> m_pipe;

   protected:
    void FeedCore(std::shared_ptr<void>& data, std::vector<size_t> const& dims) override {
        // auto outValue = Tensor<T>(std::static_pointer_cast<T>(dataShared), dims);
        m_pipe->Feed({std::static_pointer_cast<T>(data), dims});
    }

    ONNXTensorElementDataType ExpectedElementType() const override { return ::MapToOrtType<T>(); }

   public:
    Poker(std::shared_ptr<inference::InputPipe> pipe)
        : m_pipe(std::static_pointer_cast<inference::DirectInputPipe<inference::Tensor<T>>>(std::move(pipe))) {}
    ~Poker() = default;
};

/**
 * @brief Utility function for creating a typed peeker given a handle.
 *
 * @param handle The value handle for which we create the peeker.
 * @return std::unique_ptr<inference::IPeeker> A peeker instance.
 */
std::unique_ptr<IPeeker> IPeeker::CreatePeeker(std::shared_ptr<inference::IHandle> handle) noexcept {
    auto type = handle->Type();
    if (type == ::MakeTensorType<float_t>()) return std::make_unique<Peeker<float_t>>(handle);
    if (type == ::MakeTensorType<double_t>()) return std::make_unique<Peeker<double_t>>(handle);
    if (type == ::MakeTensorType<int32_t>()) return std::make_unique<Peeker<int32_t>>(handle);
    if (type == ::MakeTensorType<int64_t>()) return std::make_unique<Peeker<int64_t>>(handle);
    // Should be impossible, since by the time this function is called all types
    // would have been tested to ensure they are one of the supported types.
    return nullptr;
}

/**
 * @brief Utility function for creating a typed poker given a pipe.
 *
 * @param pipe The input pipe for which we create the poker.
 * @return std::unique_ptr<inference::IPoker> A poker instance.
 */
std::unique_ptr<IPoker> IPoker::CreatePoker(std::shared_ptr<inference::InputPipe> pipe) noexcept {
    auto type = pipe->Type();
    if (type == ::MakeTensorType<float_t>()) return std::make_unique<Poker<float_t>>(pipe);
    if (type == ::MakeTensorType<double_t>()) return std::make_unique<Poker<double_t>>(pipe);
    if (type == ::MakeTensorType<int32_t>()) return std::make_unique<Poker<int32_t>>(pipe);
    if (type == ::MakeTensorType<int64_t>()) return std::make_unique<Poker<int64_t>>(pipe);
    // Should be impossible.
    return nullptr;
}

/**
 * The state class hides away all ONNX runtime library specific structures, similarly to the model's state.
 */
class Model::UpdaterImpl::State {
   public:
    std::shared_ptr<Model::State> m_modelState;

    std::vector<std::string> m_inputNames;
    std::vector<const char*> m_inputCNames;
    std::vector<std::unique_ptr<IPeeker>> m_peekers;
    std::vector<OrtValue*> m_inputs;

    std::vector<std::string> m_outputNames;
    std::vector<const char*> m_outputCNames;
    std::vector<std::unique_ptr<IPoker>> m_pokers;

    State(onnx_errc& errc, std::shared_ptr<Model::State> modelState,
          std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
          std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe)
        : m_modelState(std::move(modelState)) {
        m_inputNames.reserve(inputToHandle.size());
        m_inputCNames.reserve(inputToHandle.size());
        m_peekers.reserve(inputToHandle.size());
        for (const auto& pair : inputToHandle) {
            m_peekers.push_back(IPeeker::CreatePeeker(pair.second));
            m_inputNames.push_back(pair.first);
            m_inputCNames.push_back(m_inputNames.back().c_str());
        }
        m_inputs.resize(inputToHandle.size());

        m_outputNames.reserve(outputToPipe.size());
        m_outputCNames.reserve(outputToPipe.size());
        m_pokers.reserve(outputToPipe.size());
        for (const auto& pair : outputToPipe) {
            m_pokers.push_back(IPoker::CreatePoker(pair.second));
            m_outputNames.push_back(pair.first);
            m_outputCNames.push_back(m_outputNames.back().c_str());
        }
    }

    ~State() {
        for (OrtValue*& value : m_inputs) {
            if (value != nullptr) OrtReleaseValue(value);
        }
    }
};

rt::expected<std::shared_ptr<inference::ValueUpdater>> Model::CreateValueUpdater(
    std::map<std::string, std::shared_ptr<inference::IHandle>> const& inputToHandle,
    std::map<std::string, std::shared_ptr<inference::InputPipe>> const& outputToPipe,
    std::function<void()> outOfBandNotifier) const {
    // Do checks on the handles and pipes before we continue.
    for (const auto& pair : inputToHandle) {
        const auto& found = m_inputs.find(pair.first);
        if (found == m_inputs.cend()) return make_onnx_unexpected(onnx_errc::unknown_input);
        if (pair.second->Type() != found->second) return make_onnx_unexpected(onnx_errc::type_mismatch);
    }
    for (const auto& pair : outputToPipe) {
        const auto& found = m_outputs.find(pair.first);
        if (found == m_outputs.cend()) return make_onnx_unexpected(onnx_errc::unknown_input);
        if (pair.second->Type() != found->second) return make_onnx_unexpected(onnx_errc::type_mismatch);
    }
    auto errc = onnx_model::onnx_errc();
    auto updaterState = std::make_unique<Model::UpdaterImpl::State>(errc, m_state, inputToHandle, outputToPipe);
    outOfBandNotifier();
    return std::shared_ptr<inference::ValueUpdater>(
        new UpdaterImpl(std::static_pointer_cast<const Model>(shared_from_this()), std::move(updaterState)));
}

Model::UpdaterImpl::UpdaterImpl(std::shared_ptr<const Model> parent, std::unique_ptr<State>&& state)
    : m_parent(std::move(parent)), m_state(std::move(state)) {}
Model::UpdaterImpl::~UpdaterImpl() {}

std::error_code Model::UpdaterImpl::UpdateOutput() {
    const onnx_errc no_error = onnx_errc();
    const OrtAllocatorInfo* allocInfo = m_parent->m_state->m_allocInfo.get();
    // Gather the inputs.
    size_t i = 0;
    for (auto& peeker : m_state->m_peekers) {
        onnx_errc errc = peeker->Peek(m_state->m_inputs[i++], allocInfo);
        if (errc != no_error) return make_onnx_error(errc);
    }

    // Run the model.

    // As far as I can tell in the current version, OrtRun's outputs is a purely *out* parameter (at least according to
    // the header), so the input pointers would not be reused even if we were to furnish them (in which case I'd expect
    // this to be an in-out parameter). But I could be misunderstanding the semantics.
    std::vector<OrtValue*> outputs;
    outputs.resize(m_state->m_outputCNames.size());
    OrtStatus* status;
    {
        // Have the call to OrtRun occur inside a critical section.
        std::unique_lock<std::mutex> lock(m_state->m_modelState->m_model_lock);
        status = OrtRun(m_parent->m_state->m_session.get(), nullptr, m_state->m_inputCNames.data(),
                        m_state->m_inputs.data(), m_state->m_inputs.size(), m_state->m_outputCNames.data(),
                        m_state->m_outputCNames.size(), outputs.data());
    }
    if (status) {
        OrtReleaseStatus(status);
        return make_onnx_error(onnx_errc::run_error);
    }

    // Gather the outputs.
    i = 0;
    for (auto& poker : m_state->m_pokers) {
        poker->Poke(outputs.at(i++), allocInfo);
    }

    return std::error_code();
}
}  // namespace onnx_model
