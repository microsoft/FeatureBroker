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
    env.hpp
    env.cpp
    model_test.cpp
    raw_slim.cpp
    main.cpp)

set(VWSLIM_TEST_DATA_DIR ${resonance_EXTERNAL}/vw/vowpalwabbit/slim/test/data)

foreach (localTarget ${FB_TEST} ${FB_TEST_STATIC})
    add_executable(${localTarget} ${TEST_SRC})
    set_target_properties(${localTarget} PROPERTIES FOLDER "Test")
    add_custom_command(TARGET ${localTarget} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${VWSLIM_TEST_DATA_DIR}
            $<TARGET_FILE_DIR:${localTarget}>/slimdata)
    target_link_libraries(${localTarget} PRIVATE gtest gtest_main)
    # Temporarily link ${VWSLIM_LIB_NAME} with the relevant flags for direct invocation for now.
    target_link_libraries(${localTarget} PRIVATE ${VWSLIM_LIB_NAME})
endforeach(localTarget)

target_link_libraries(${FB_TEST_STATIC} PRIVATE ${LIB_NAME}_static)
target_link_libraries(${FB_TEST} PRIVATE ${LIB_NAME})

if(MSVC)
    # The dynamic library test needs the DLL copied. Cribbed from answers here:
    # https://stackoverflow.com/questions/13429656/how-to-copy-contents-of-a-directory-into-build-directory-after-make-with-cmake
    set(dllDir ${CMAKE_BINARY_DIR}/src/${LIB_NAME}/$<CONFIG>)
    set(coreDllDir ${CMAKE_BINARY_DIR}/src/${CORE_LIB_NAME}/$<CONFIG>)

    add_custom_command(TARGET ${FB_TEST} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${dllDir}
            $<TARGET_FILE_DIR:${FB_TEST}>)

    add_custom_command(TARGET ${FB_TEST} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${coreDllDir}
            $<TARGET_FILE_DIR:${FB_TEST}>)
endif(MSVC)

# All the tests need the files.

gtest_discover_tests(${FB_TEST})
gtest_discover_tests(${FB_TEST_STATIC})
