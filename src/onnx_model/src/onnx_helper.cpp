/*

#include <auf/auf_log2_api.hpp>
#include <rt/rt_spinlock.hpp>
#include <spl/spl_file.hpp>
#include <spl/spl_init.hpp>

#include "private/onnx/onnx_dll_helper.hpp"

namespace inference {
AUF_LOG_DECL_COMPONENT_NS();
}
AUF_LOG_USE_COMPONENT_NS(inference);

#if defined(SPL_PLATFORM_WINDOWS)
constexpr auto OnnxDllExt = "dll";
constexpr auto OnnxDllName = "onnxruntime";
#elif defined(SPL_PLATFORM_MACOSX)
constexpr auto OnnxDllExt = "dylib";
constexpr auto OnnxDllName = "libonnxruntime";
#else
constexpr auto OnnxDllExt = "so";
constexpr auto OnnxDllName = "onnxruntime";
#endif

static spl::dynlib_name_t getOnnxDllPath() {
    static spl::dynlib_name_t libPath = []() -> spl::dynlib_name_t {
        spl::Path path;
        spl::pathInitFromLocation(path, spl::PL_MODULE_DIR);
        spl::pathAppendComponent(path, OnnxDllName, OnnxDllExt);
        static spl::PathCharType buffer[SPL_MAX_PATH];
#if defined(SPL_PLATFORM_WINDOWS)
        if (!path.isGood() || spl::wcsncpy_s(buffer, path.wcsValue(), path.capacity())) {
            return L"onnxruntime.dll";
        }
#else
        if (!path.isGood() ||
            spl::strncpy_s(buffer, path.stringValue(), spl::strnlen_s(path.stringValue(), SPL_MAX_PATH))) {
#if defined(SPL_PLATFORM_MACOSX)
            return "libonnxruntime.dylib";
#else
            return "onnxruntime.so";
#endif
        }
#endif
        return buffer;
    }();
    return libPath;
}

inference::onnx::DllHelper::DllHelper() noexcept
    : OrtGetErrorCode{getOnnxDllPath(), "OrtGetErrorCode"},
      OrtReleaseEnv{OrtGetErrorCode, "OrtReleaseEnv"},
      OrtReleaseStatus{OrtGetErrorCode, "OrtReleaseStatus"},
      OrtReleaseAllocator{OrtGetErrorCode, "OrtReleaseAllocator"},
      OrtReleaseAllocatorInfo{OrtGetErrorCode, "OrtReleaseAllocatorInfo"},
      OrtReleaseSession{OrtGetErrorCode, "OrtReleaseSession"},
      OrtReleaseTypeInfo{OrtGetErrorCode, "OrtReleaseTypeInfo"},
      OrtReleaseTensorTypeAndShapeInfo{OrtGetErrorCode, "OrtReleaseTensorTypeAndShapeInfo"},
      OrtReleaseSessionOptions{OrtGetErrorCode, "OrtReleaseSessionOptions"},
      OrtReleaseValue{OrtGetErrorCode, "OrtReleaseValue"},
      OrtCreateDefaultAllocator{OrtGetErrorCode, "OrtCreateDefaultAllocator"},
      OrtCreateCpuAllocatorInfo{OrtGetErrorCode, "OrtCreateCpuAllocatorInfo"},
      OrtAllocatorGetInfo{OrtGetErrorCode, "OrtAllocatorGetInfo"},
      OrtCastTypeInfoToTensorInfo{OrtGetErrorCode, "OrtCastTypeInfoToTensorInfo"},
      OrtGetTensorElementType{OrtGetErrorCode, "OrtGetTensorElementType"},
      OrtGetDimensionsCount{OrtGetErrorCode, "OrtGetDimensionsCount"},
      OrtGetDimensions{OrtGetErrorCode, "OrtGetDimensions"},
      OrtCreateTensorWithDataAsOrtValue{OrtGetErrorCode, "OrtCreateTensorWithDataAsOrtValue"},
      OrtCreateTensorAsOrtValue{OrtGetErrorCode, "OrtCreateTensorAsOrtValue"},
      OrtIsTensor{OrtGetErrorCode, "OrtIsTensor"},
      OrtGetValueType{OrtGetErrorCode, "OrtGetValueType"},
      OrtGetTypeInfo{OrtGetErrorCode, "OrtGetTypeInfo"},
      OrtOnnxTypeFromTypeInfo{OrtGetErrorCode, "OrtOnnxTypeFromTypeInfo"},
      OrtGetTensorMutableData{OrtGetErrorCode, "OrtGetTensorMutableData"},
      OrtCreateEnv{OrtGetErrorCode, "OrtCreateEnv"},
      OrtCreateEnvWithCustomLogger{OrtGetErrorCode, "OrtCreateEnvWithCustomLogger"},
      OrtCreateSessionOptions{OrtGetErrorCode, "OrtCreateSessionOptions"},
      OrtEnableCpuMemArena{OrtGetErrorCode, "OrtEnableCpuMemArena"},
      OrtDisableCpuMemArena{OrtGetErrorCode, "OrtDisableCpuMemArena"},
      OrtEnableSequentialExecution{OrtGetErrorCode, "OrtEnableSequentialExecution"},
      OrtDisableSequentialExecution{OrtGetErrorCode, "OrtDisableSequentialExecution"},
      OrtSetSessionThreadPoolSize{OrtGetErrorCode, "OrtSetSessionThreadPoolSize"},
      OrtSetSessionGraphOptimizationLevel{OrtGetErrorCode, "OrtSetSessionGraphOptimizationLevel"},
      OrtSetSessionLogVerbosityLevel{OrtGetErrorCode, "OrtSetSessionLogVerbosityLevel"},
      OrtCreateSession{OrtGetErrorCode, "OrtCreateSession"},
      OrtCreateSessionFromArray{OrtGetErrorCode, "OrtCreateSessionFromArray"},
      OrtSessionGetInputCount{OrtGetErrorCode, "OrtSessionGetInputCount"},
      OrtSessionGetOutputCount{OrtGetErrorCode, "OrtSessionGetOutputCount"},
      OrtSessionGetInputName{OrtGetErrorCode, "OrtSessionGetInputName"},
      OrtSessionGetOutputName{OrtGetErrorCode, "OrtSessionGetOutputName"},
      OrtSessionGetInputTypeInfo{OrtGetErrorCode, "OrtSessionGetInputTypeInfo"},
      OrtSessionGetOutputTypeInfo{OrtGetErrorCode, "OrtSessionGetOutputTypeInfo"},
      OrtRun{OrtGetErrorCode, "OrtRun"} {}

std::shared_ptr<inference::onnx::DllHelper> inference::onnx::DllHelper::get() noexcept {
    static volatile int spinlock;
    static std::shared_ptr<DllHelper>* ptr;
    {
        rt::ScopedSharedSpinlock lock(spinlock);
        if (ptr) return *ptr;
    }

    rt::ScopedUniqueSpinlock lock(spinlock);
    if (!ptr) {
        ptr = new std::shared_ptr<DllHelper>(std::make_shared<DllHelper>());

        // ptr = new std::shared_ptr<DllHelper>(new DllHelper(), [](DllHelper* c) {});

        AUF_LOG_ABORT_IF_NOT(ptr);
        spl::atStop("inference.onnx.DllHelper", [] {
            rt::ScopedUniqueSpinlock lock(spinlock);
            if (ptr) {
                delete ptr;
                ptr = nullptr;
            }
        });
    }
    return *ptr;
}

*/
