cmake_minimum_required(VERSION 3.30)

# -------------------------------------
# ------- Project Configuration -------
# -------------------------------------

project(CInfinity 
    VERSION ${Infinity_VERSION}
    DESCRIPTION "A C++20 graph-transformer based chess bot."
    HOMEPAGE_URL "https://github.com/KhrySystemAI/CInfinity" 
    LANGUAGES CXX
)

configure_file(
    include/cinfinity/config.hpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cinfinity/config.hpp
    @ONLY
)

add_library(cinfinity STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transposition_table.cpp"
)

target_include_directories(cinfinity PUBLIC 
    ${ONNXRUNTIME_INCLUDE_DIR}
    "include"
    ${chess_SOURCE_DIR}/include
    ${uint128_t_SOURCE_DIR}
)
target_link_libraries(cinfinity PUBLIC 
    onnxruntime
    unordered_dense::unordered_dense
    pybind11::headers
)
if(${INFINITY_COMPILE_TESTS})
    include(GoogleTest)

    enable_testing()

    add_executable(
        test_transposition_table
        tests/test_transposition_table.cpp
    )

    target_link_libraries(test_transposition_table PRIVATE 
        cinfinity
        GTest::gtest_main
    )
    gtest_discover_tests(test_transposition_table)
endif()

if(${INFINITY_COMPILE_FUZZING})
    add_executable(
        fuzz_transposition_table 
        tests/fuzz_transposition_table.cpp
    )

    target_link_libraries(fuzz_transposition_table PRIVATE 
        cinfinity
        GTest::gtest_main
    )
    link_fuzztest(fuzz_transposition_table)
    gtest_discover_tests(fuzz_transposition_table)
endif()