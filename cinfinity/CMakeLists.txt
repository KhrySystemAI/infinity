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

add_library(cinfinity SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transposition_table.cpp"
)

target_include_directories(cinfinity PUBLIC 
    "include"
    ${chess_SOURCE_DIR}/include
    ${uint128_t_SOURCE_DIR}
)
target_link_libraries(cinfinity PUBLIC 
    unordered_dense::unordered_dense
    pybind11::headers
)