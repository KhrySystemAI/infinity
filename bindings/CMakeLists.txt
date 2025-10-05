cmake_minimum_required(VERSION 3.30)

find_package(pybind11 REQUIRED CONFIG)
find_package(Sanitizers REQUIRED)

add_library(cinfinity_bindings_cxx STATIC
    src/cinfinity_cxx_bindings.cpp
)
add_sanitizers(cinfinity_bindings_cxx)

target_link_libraries(cinfinity_bindings_cxx PRIVATE
    cinfinity::core
)

add_library(cinfinity::bindings::cxx ALIAS cinfinity_bindings_cxx)