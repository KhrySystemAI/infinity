cmake_minimum_required(VERSION 3.30)

set(_cinfinity_core_hpp 
    include/cinfinity/core/queue_builder.hpp
)

add_library(cinfinity_core INTERFACE
    ${_cinfinity_core_hpp}
)

target_precompile_headers(cinfinity_core INTERFACE
    ${_cinfinity_core_hpp}
)

target_include_directories(cinfinity_core INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cinfinity_core INTERFACE
    cinfinity::memory
    cinfinity::chess
)

add_library(cinfinity::core ALIAS cinfinity_core)