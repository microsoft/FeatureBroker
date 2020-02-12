cmake_minimum_required(VERSION 3.16)

include(GoogleTest)

include_directories(
        ${resonance_EXTERNAL}/gtest/googletest/include
        $<TARGET_PROPERTY:${CORE_LIB_NAME},INCLUDE_DIRECTORIES>)

set(ONNX_TEST ${LIB_NAME}_test)
set(ONNX_TEST_STATIC ${LIB_NAME}_stest)

set(TEST_SRC
    env.cpp
    env.hpp
    main.cpp
    model_test.cpp)

foreach (localTarget ${ONNX_TEST} ${ONNX_TEST_STATIC})
    add_executable(${localTarget} ${TEST_SRC})
    set_target_properties(${localTarget} PROPERTIES FOLDER "Test")
    target_link_libraries(${localTarget} PRIVATE gtest gtest_main)
endforeach(localTarget)

target_link_libraries(${ONNX_TEST_STATIC} PUBLIC ${LIB_NAME}_static)
target_link_libraries(${ONNX_TEST} PUBLIC ${LIB_NAME})

if(MSVC)
    # The dynamic library test needs the DLL copied. Cribbed from answers here:
    # https://stackoverflow.com/questions/13429656/how-to-copy-contents-of-a-directory-into-build-directory-after-make-with-cmake
    set(dllDir ${CMAKE_BINARY_DIR}/src/${LIB_NAME}/$<CONFIG>)
    set(coreDllDir ${CMAKE_BINARY_DIR}/src/${CORE_LIB_NAME}/$<CONFIG>)

    add_custom_command(TARGET ${ONNX_TEST} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${dllDir}
            $<TARGET_FILE_DIR:${ONNX_TEST}>
            DEPENDS onnx_model)
    
    add_custom_command(TARGET ${ONNX_TEST} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${coreDllDir}
            $<TARGET_FILE_DIR:${ONNX_TEST}>
            DEPENDS onnx_model)

    add_custom_command(TARGET ${ONNX_TEST} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${ONNX_LIB_DIR}
            $<TARGET_FILE_DIR:${ONNX_TEST}>
            DEPENDS onnx_model)
endif(MSVC)

message(${CMAKE_CURRENT_SOURCE_DIR})
message(${CMAKE_CURRENT_BINARY_DIR})
add_custom_command(TARGET ${ONNX_TEST} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul.onnx
        $<TARGET_FILE_DIR:${ONNX_TEST}>)

# All the tests need the files.

gtest_discover_tests(${ONNX_TEST})
gtest_discover_tests(${ONNX_TEST_STATIC})

