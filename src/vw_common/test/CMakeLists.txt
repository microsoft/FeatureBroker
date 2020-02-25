# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.16)

include(GoogleTest)

include_directories(
        ${resonance_EXTERNAL}/gtest/googletest/include
        $<TARGET_PROPERTY:${CORE_LIB_NAME},INCLUDE_DIRECTORIES>)

set(FB_TEST ${LIB_NAME}_test)

set(TEST_SRC
    actions_test.cpp
    env.hpp
    env.cpp
    error_test.cpp
    schema_builder_test.cpp
    main.cpp)

add_executable(${FB_TEST} ${TEST_SRC})
set_target_properties(${FB_TEST} PROPERTIES FOLDER "Test")
target_link_libraries(${FB_TEST} PRIVATE gtest gtest_main)

target_link_libraries(${FB_TEST} PRIVATE ${LIB_NAME})

# All the tests need the files.
gtest_discover_tests(${FB_TEST})
