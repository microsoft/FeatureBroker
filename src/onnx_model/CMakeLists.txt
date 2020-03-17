include (ExternalProject)

set(ONNX_VERSION 0.5.0)

SET(ONNX_RUNTIME_FILE "")
SET(ONNX_INC_DIR ${resonance_EXTERNAL}/ONNXRuntime/include)
SET(ONNX_LIB_DIR ${resonance_EXTERNAL}/ONNXRuntime/lib)
if (CMAKE_SYSTEM_NAME MATCHES Windows)
SET(ONNX_RUNTIME_LIB ${ONNX_LIB_DIR}/onnxruntime.lib)
SET(ONNX_RUNTIME_FILE onnxruntime-win-x64-${ONNX_VERSION}.zip)
elseif (CMAKE_SYSTEM_NAME MATCHES Darwin)
SET(ONNX_RUNTIME_LIB ${ONNX_LIB_DIR}/libonnxruntime.dylib)
SET(ONNX_RUNTIME_FILE onnxruntime-osx-x64-${ONNX_VERSION}.tgz)
elseif (CMAKE_SYSTEM_NAME MATCHES Linux)
SET(ONNX_RUNTIME_LIB ${ONNX_LIB_DIR}/libonnxruntime.so)
SET(ONNX_RUNTIME_FILE onnxruntime-linux-x64-${ONNX_VERSION}.tgz)
endif(CMAKE_SYSTEM_NAME MATCHES Windows)

SET(ONNX_RUNTIME_URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_RUNTIME_FILE})

ExternalProject_Add(ONNXRuntime 
   URL ${ONNX_RUNTIME_URL}
   BUILD_IN_SOURCE 1
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ./
   ${resonance_EXTERNAL}/ONNXRuntime
)

set(LIB_NAME onnx_model)
set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_model)
set(PRIVATEINCDIR ${CMAKE_CURRENT_SOURCE_DIR}/private)

set (SRC
    ${INCDIR}/onnx_error.hpp
    ${INCDIR}/model.hpp

    ${SRCDIR}/onnx_error.cpp
    ${SRCDIR}/model.cpp
)

add_library(${LIB_NAME} SHARED)
add_library(${LIB_NAME}_static STATIC)

foreach (localTarget ${LIB_NAME} ${LIB_NAME}_static)
    add_dependencies(${localTarget} ONNXRuntime)
    target_sources(${localTarget} PRIVATE ${SRC})
    target_include_directories(${localTarget} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${resonance_EXTERNAL}/expected/include
        ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private
        ${ONNX_INC_DIR})
    set_target_properties(${localTarget} PROPERTIES
        LINKER_LANGUAGE CXX
        FOLDER "Library")
endforeach(localTarget)

target_compile_definitions(${LIB_NAME}_static PUBLIC ONNX_MODEL_STATIC_DEFINE)

target_link_libraries(${LIB_NAME} PUBLIC ${CORE_LIB_NAME} ${ONNX_RUNTIME_LIB})
target_link_libraries(${LIB_NAME}_static PUBLIC ${CORE_LIB_NAME}_static ${ONNX_RUNTIME_LIB})

include(GenerateExportHeader)
generate_export_header(${LIB_NAME})

add_subdirectory(test)

