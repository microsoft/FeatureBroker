cmake_minimum_required(VERSION 3.10)
set(RESONANCE_VERSION 0.1.0)
# In order for the MSVC runtime library settings to have effect.
cmake_policy(SET CMP0091 NEW)

project(Resonance VERSION ${RESONANCE_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

include(CTest)
include(GoogleTest)
enable_testing()

# External code from submodules.

add_subdirectory(cmake/external/gtest)
# While we include tl::expected as a submodule, we only need its header.

set_target_properties(gmock PROPERTIES FOLDER "External/GTest")
set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")

add_subdirectory(src/feature_broker)
