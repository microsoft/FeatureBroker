# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Despite the fact that this include library is only a private library to the
# model shared library, building it still complains on Linux 

# The following is the actual relevant matter.
set(LIB_NAME vw_common)

set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCDIR ${CMAKE_CURRENT_SOURCE_DIR}/include/vw_common)
set(PRIVATEINCDIR ${CMAKE_CURRENT_SOURCE_DIR}/private)

set (SRC
    ${INCDIR}/actions.hpp
    ${INCDIR}/schema_builder.hpp
    ${INCDIR}/error.hpp
    ${INCDIR}/schema_entry.hpp

    ${SRCDIR}/actions.cpp
    ${SRCDIR}/schema_builder.cpp
    ${SRCDIR}/schema_entry.cpp
    ${SRCDIR}/error.cpp
)

add_library(${LIB_NAME} STATIC)
target_sources(${LIB_NAME} PRIVATE ${SRC})
target_include_directories(${LIB_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${resonance_EXTERNAL}/expected/include
	${CMAKE_CURRENT_BINARY_DIR}
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private)
set_target_properties(${LIB_NAME} PROPERTIES
	LINKER_LANGUAGE CXX
	FOLDER "Library")

target_compile_definitions(${LIB_NAME} PUBLIC VW_STATIC_DEFINE)

include(GenerateExportHeader)
generate_export_header(${LIB_NAME})

add_subdirectory(test)
