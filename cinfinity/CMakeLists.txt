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

add_subdirectory(core)
add_subdirectory(tests)