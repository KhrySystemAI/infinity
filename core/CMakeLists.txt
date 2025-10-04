cmake_minimum_required(VERSION 3.30)

find_package(absl REQUIRED CONFIG)

add_library(cinfinity_core INTERFACE
    include/cinfinity/core/transposition_table.hpp
    include/cinfinity/core/transposition_table.inl
)

target_precompile_headers(cinfinity_core INTERFACE 
    include/cinfinity/core/transposition_table.hpp
)

target_include_directories(cinfinity_core INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cinfinity_core INTERFACE 
    absl::flat_hash_map
    absl::random_random
    absl::synchronization
)

add_library(cinfinity::core ALIAS cinfinity_core)