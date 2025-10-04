cmake_minimum_required(VERSION 3.30)

add_library(cinfinity_bindings_cxx STATIC
    cinfinity_cxx_bindings.cpp
)

target_link_libraries(cinfinity_bindings_cxx PRIVATE
    cinfinity::core
)

add_library(cinfinity::bindings::cxx ALIAS cinfinity_bindings_cxx)