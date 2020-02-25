# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Either set up the build of that library ourselves, or utilize that as defined through the CMake files.
set(LOCAL_VWSLIM FALSE)

if(LOCAL_VWSLIM)
    set(VWSLIM_LIB_NAME my_slim_vw)
    set(VW_DIR ${resonance_EXTERNAL}/vw)
    set(VWSLIM_SRC_DIR ${VW_DIR}/vowpalwabbit/slim/src)

    add_library(${VWSLIM_LIB_NAME} STATIC
        ${VWSLIM_SRC_DIR}/example_predict_builder.cc
        ${VWSLIM_SRC_DIR}/model_parser.cc
        ${VWSLIM_SRC_DIR}/opts.cc
        ${VWSLIM_SRC_DIR}/vw_slim_predict.cc
        ${VWSLIM_SRC_DIR}/../../example_predict.cc)

    target_compile_definitions(${VWSLIM_LIB_NAME} PUBLIC EXPLORE_NOEXCEPT VW_NOEXCEPT)

    target_include_directories(${VWSLIM_LIB_NAME} PUBLIC
        ${VW_DIR}/vowpalwabbit/slim/include
        ${VW_DIR}/vowpalwabbit
        ${VW_DIR}/explore)
else(LOCAL_VWSLIM)
    set(VWSLIM_LIB_NAME vwslim)
endif(LOCAL_VWSLIM)

set_target_properties(${VWSLIM_LIB_NAME} PROPERTIES
    FOLDER "External/VW"
    POSITION_INDEPENDENT_CODE TRUE)

# Despite the fact that this include library is only a private library to the
# model shared library, building it still complains on Linux 

# The following is the actual relevant matter.
set(LIB_NAME vw_slim_model)

set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/include/vw_slim_model)
set(PRIVATEINCDIR ${CMAKE_CURRENT_SOURCE_DIR}/private)

set (SRC
    ${INCDIR}/model.hpp
    ${SRCDIR}/model.cpp
)

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
    target_link_libraries(${localTarget} PRIVATE ${VWSLIM_LIB_NAME})
endforeach(localTarget)

include(GenerateExportHeader)
generate_export_header(${LIB_NAME})

target_compile_definitions(${LIB_NAME}_static PUBLIC VW_SLIM_MODEL_STATIC_DEFINE)

target_link_libraries(${LIB_NAME} PUBLIC ${CORE_LIB_NAME} vw_common)
target_link_libraries(${LIB_NAME}_static PUBLIC ${CORE_LIB_NAME}_static vw_common)

add_subdirectory(test)
