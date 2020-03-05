# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set(REMOTE_VWSLIM FALSE)

# The following is the actual relevant matter.
set(LIB_NAME vw_remote_model)
set(VCPKG_TARGET_TRIPLET CACHE STRING ${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}-windows-static-md)
find_package(cpprestsdk CONFIG REQUIRED)

set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/include/vw_remote_model)
set(PRIVATEINCDIR ${CMAKE_CURRENT_SOURCE_DIR}/private)

set (SRC
    ${INCDIR}/cpprest_recommender_client.hpp
    ${INCDIR}/error.hpp
    ${INCDIR}/irecommender_client.hpp
    ${INCDIR}/remote_model.hpp
    ${SRCDIR}/cpprest_recommender_client.cpp
    ${SRCDIR}/error.cpp
    ${SRCDIR}/remote_model.cpp
)

add_library(${LIB_NAME} SHARED)
add_library(${LIB_NAME}_static STATIC)

foreach (localTarget ${LIB_NAME} ${LIB_NAME}_static)
    target_sources(${localTarget} PRIVATE ${SRC})
    target_include_directories(${localTarget} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../feature_broker/include
        ${resonance_EXTERNAL}/expected/include
        ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private)
    set_target_properties(${localTarget} PROPERTIES
        LINKER_LANGUAGE CXX
        FOLDER "Library")
endforeach(localTarget)

target_link_libraries(${LIB_NAME} PUBLIC ${CORE_LIB_NAME} vw_common cpprestsdk::cpprest cpprestsdk::cpprestsdk_zlib_internal)
target_link_libraries(${LIB_NAME}_static PUBLIC ${CORE_LIB_NAME} vw_common cpprestsdk::cpprest cpprestsdk::cpprestsdk_zlib_internal)

include(GenerateExportHeader)
generate_export_header(${LIB_NAME})

target_compile_definitions(${LIB_NAME}_static PUBLIC VW_REMOTE_MODEL_STATIC_DEFINE)
add_subdirectory(test)
