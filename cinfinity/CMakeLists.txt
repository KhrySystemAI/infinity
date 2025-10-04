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

add_subdirectory(core)