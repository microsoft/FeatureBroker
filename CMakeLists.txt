# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)
set(RESONANCE_VERSION 0.1.0)
# In order for the MSVC runtime library settings to have effect.
cmake_policy(SET CMP0091 NEW)

project(Resonance VERSION ${RESONANCE_VERSION} LANGUAGES CXX)
set(CORE_LIB_NAME feature_broker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set (resonance_EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/cmake/external)

include(CTest)
include(GoogleTest)
enable_testing()

# External code from submodules.

add_subdirectory(${resonance_EXTERNAL}/gtest)
# While we include tl::expected as a submodule, we only need its header.

set_target_properties(gmock PROPERTIES FOLDER "External/GTest")
set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")

add_subdirectory(src/feature_broker)

# For Vowpal Wabbit Slim. Note the two isolated includes, as opposed to adding
# the root directory itself. This is due to some odd requirements that VW slim
# itself does not need being enforced at the root level of the project, as opposed
# to merely on those components that actually make use of those requirements. VW
# Slim by itself compiles just fine, but we do have to be somewhat selective about
# the specific directories we include.
add_subdirectory(${resonance_EXTERNAL}/vw/explore)
add_subdirectory(${resonance_EXTERNAL}/vw/vowpalwabbit/slim)
add_subdirectory(src/vw_common)
add_subdirectory(src/vw_slim_model)
add_subdirectory(src/onnx_model)
