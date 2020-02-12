/*
#pragma once
#include <onnxruntime_c_api.h>

#include <auf/auf_dynamic_function.hpp>
#include <memory>

namespace inference {
namespace onnx {
struct DllHelper {
    static std::shared_ptr<DllHelper> get() noexcept;

    DllHelper() noexcept;

    auf::DynamicFunction<decltype(&::OrtGetErrorCode)> OrtGetErrorCode;

    auf::DynamicFunctionView<decltype(&::OrtReleaseEnv)> OrtReleaseEnv;
    auf::DynamicFunctionView<decltype(&::OrtReleaseStatus)> OrtReleaseStatus;
    auf::DynamicFunctionView<decltype(&::OrtReleaseAllocator)> OrtReleaseAllocator;
    auf::DynamicFunctionView<decltype(&::OrtReleaseAllocatorInfo)> OrtReleaseAllocatorInfo;
    auf::DynamicFunctionView<decltype(&::OrtReleaseSession)> OrtReleaseSession;
    auf::DynamicFunctionView<decltype(&::OrtReleaseTypeInfo)> OrtReleaseTypeInfo;
    auf::DynamicFunctionView<decltype(&::OrtReleaseTensorTypeAndShapeInfo)> OrtReleaseTensorTypeAndShapeInfo;
    auf::DynamicFunctionView<decltype(&::OrtReleaseSessionOptions)> OrtReleaseSessionOptions;
    auf::DynamicFunctionView<decltype(&::OrtReleaseValue)> OrtReleaseValue;

    auf::DynamicFunctionView<decltype(&::OrtCreateDefaultAllocator)> OrtCreateDefaultAllocator;
    auf::DynamicFunctionView<decltype(&::OrtCreateCpuAllocatorInfo)> OrtCreateCpuAllocatorInfo;
    auf::DynamicFunctionView<decltype(&::OrtAllocatorGetInfo)> OrtAllocatorGetInfo;
    auf::DynamicFunctionView<decltype(&::OrtCastTypeInfoToTensorInfo)> OrtCastTypeInfoToTensorInfo;
    auf::DynamicFunctionView<decltype(&::OrtGetTensorElementType)> OrtGetTensorElementType;
    auf::DynamicFunctionView<decltype(&::OrtGetDimensionsCount)> OrtGetDimensionsCount;
    auf::DynamicFunctionView<decltype(&::OrtGetDimensions)> OrtGetDimensions;
    auf::DynamicFunctionView<decltype(&::OrtCreateTensorWithDataAsOrtValue)> OrtCreateTensorWithDataAsOrtValue;
    auf::DynamicFunctionView<decltype(&::OrtCreateTensorAsOrtValue)> OrtCreateTensorAsOrtValue;
    auf::DynamicFunctionView<decltype(&::OrtIsTensor)> OrtIsTensor;
    auf::DynamicFunctionView<decltype(&::OrtGetValueType)> OrtGetValueType;
    auf::DynamicFunctionView<decltype(&::OrtGetTypeInfo)> OrtGetTypeInfo;
    auf::DynamicFunctionView<decltype(&::OrtOnnxTypeFromTypeInfo)> OrtOnnxTypeFromTypeInfo;
    auf::DynamicFunctionView<decltype(&::OrtGetTensorMutableData)> OrtGetTensorMutableData;
    auf::DynamicFunctionView<decltype(&::OrtCreateEnv)> OrtCreateEnv;
    auf::DynamicFunctionView<decltype(&::OrtCreateEnvWithCustomLogger)> OrtCreateEnvWithCustomLogger;
    auf::DynamicFunctionView<decltype(&::OrtCreateSessionOptions)> OrtCreateSessionOptions;
    auf::DynamicFunctionView<decltype(&::OrtEnableCpuMemArena)> OrtEnableCpuMemArena;
    auf::DynamicFunctionView<decltype(&::OrtDisableCpuMemArena)> OrtDisableCpuMemArena;
    auf::DynamicFunctionView<decltype(&::OrtEnableSequentialExecution)> OrtEnableSequentialExecution;
    auf::DynamicFunctionView<decltype(&::OrtDisableSequentialExecution)> OrtDisableSequentialExecution;
    auf::DynamicFunctionView<decltype(&::OrtSetSessionThreadPoolSize)> OrtSetSessionThreadPoolSize;
    auf::DynamicFunctionView<decltype(&::OrtSetSessionGraphOptimizationLevel)> OrtSetSessionGraphOptimizationLevel;
    auf::DynamicFunctionView<decltype(&::OrtSetSessionLogVerbosityLevel)> OrtSetSessionLogVerbosityLevel;
    auf::DynamicFunctionView<decltype(&::OrtCreateSession)> OrtCreateSession;
    auf::DynamicFunctionView<decltype(&::OrtCreateSessionFromArray)> OrtCreateSessionFromArray;
    auf::DynamicFunctionView<decltype(&::OrtSessionGetInputCount)> OrtSessionGetInputCount;
    auf::DynamicFunctionView<decltype(&::OrtSessionGetOutputCount)> OrtSessionGetOutputCount;
    auf::DynamicFunctionView<decltype(&::OrtSessionGetInputName)> OrtSessionGetInputName;
    auf::DynamicFunctionView<decltype(&::OrtSessionGetOutputName)> OrtSessionGetOutputName;
    auf::DynamicFunctionView<decltype(&::OrtSessionGetInputTypeInfo)> OrtSessionGetInputTypeInfo;
    auf::DynamicFunctionView<decltype(&::OrtSessionGetOutputTypeInfo)> OrtSessionGetOutputTypeInfo;
    auf::DynamicFunctionView<decltype(&::OrtRun)> OrtRun;

   private:
    DllHelper(const DllHelper&) = delete;
    DllHelper& operator=(const DllHelper&) = delete;
};

}  // namespace onnx
}  // namespace inference

*/
