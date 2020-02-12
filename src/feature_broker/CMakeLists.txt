# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(LIB_NAME ${CORE_LIB_NAME})
set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/include/inference)

set (SRC
    ${INCDIR}/direct_input_pipe.hpp
    ${INCDIR}/feature_broker.hpp
    ${INCDIR}/feature_broker_base.hpp
    ${INCDIR}/feature_error.hpp
    ${INCDIR}/feature_provider.hpp
    ${INCDIR}/handle.hpp
    ${INCDIR}/input_pipe.hpp
    ${INCDIR}/model.hpp
    ${INCDIR}/output_pipe.hpp
    ${INCDIR}/output_pipe_with_input.hpp
    ${INCDIR}/synchronous_feature_broker.hpp
    ${INCDIR}/tensor.hpp
    ${INCDIR}/type_descriptor.hpp
    ${INCDIR}/value_updater.hpp

    ${SRCDIR}/tensor.cpp
    ${SRCDIR}/feature_broker.cpp
    ${SRCDIR}/feature_broker_base.cpp
    ${SRCDIR}/feature_error.cpp
    ${SRCDIR}/input_pipe.cpp
    ${SRCDIR}/synchronous_feature_broker.cpp)

add_library(${LIB_NAME} SHARED)
add_library(${LIB_NAME}_static STATIC)
    
foreach (localTarget ${LIB_NAME} ${LIB_NAME}_static)
    target_sources(${localTarget} PRIVATE ${SRC})
    target_include_directories(${localTarget} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${resonance_EXTERNAL}/expected/include
        ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${localTarget} PROPERTIES
        LINKER_LANGUAGE CXX
        FOLDER "Library")
endforeach(localTarget)

target_compile_definitions(${LIB_NAME}_static PUBLIC FEATURE_BROKER_STATIC_DEFINE)

include(GenerateExportHeader)
generate_export_header(${LIB_NAME})

add_subdirectory(test)
