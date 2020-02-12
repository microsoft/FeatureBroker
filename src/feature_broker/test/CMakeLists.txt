# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.16)

include(GoogleTest)

include_directories(
        ${resonance_EXTERNAL}/gtest/googletest/include
        $<TARGET_PROPERTY:${CORE_LIB_NAME},INCLUDE_DIRECTORIES>)

set(FB_TEST ${LIB_NAME}_test)
set(FB_TEST_STATIC ${LIB_NAME}_stest)

set(TEST_SRC
    main.cpp

    add_five_model.hpp
    add_model.hpp
    error_model.hpp
    release_model.hpp
    three_output_model.hpp
    tuple_feature_providers.hpp

    feature_broker_test.cpp
    feature_provider_test.cpp
    multi_output_test.cpp
    multithread_test.cpp
    type_descriptor_test.cpp)

add_executable(${FB_TEST_STATIC} ${TEST_SRC})
add_executable(${FB_TEST} ${TEST_SRC})

target_link_libraries(${FB_TEST_STATIC} gtest gtest_main ${LIB_NAME}_static)
target_link_libraries(${FB_TEST} gtest gtest_main ${LIB_NAME})

set_target_properties(${FB_TEST} PROPERTIES FOLDER "Test")
set_target_properties(${FB_TEST_STATIC} PROPERTIES FOLDER "Test")

if(MSVC)
    # The dynamic library test needs the DLL copied. Cribbed from answers here:
    # https://stackoverflow.com/questions/13429656/how-to-copy-contents-of-a-directory-into-build-directory-after-make-with-cmake
    set(dllDir ${CMAKE_BINARY_DIR}/src/${LIB_NAME}/$<CONFIG>)

    add_custom_command(TARGET ${LIB_NAME}_test PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${dllDir}
            $<TARGET_FILE_DIR:${LIB_NAME}_test>)
endif(MSVC)

gtest_discover_tests(${LIB_NAME}_stest)
gtest_discover_tests(${LIB_NAME}_test)
