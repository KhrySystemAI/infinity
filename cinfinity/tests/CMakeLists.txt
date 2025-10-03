cmake_minimum_required(VERSION 3.30)

if(${INFINITY_COMPILE_TESTS})
    include(CTest)
    find_package(GTest REQUIRED CONFIG)
    include(GoogleTest)

    enable_testing()

    add_executable(
        test_transposition_table
        test_transposition_table.cpp
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
        fuzz_transposition_table.cpp
    )

    target_link_libraries(fuzz_transposition_table PRIVATE 
        cinfinity
        GTest::gtest_main
    )
    link_fuzztest(fuzz_transposition_table)
    gtest_discover_tests(fuzz_transposition_table)
endif()