cmake_minimum_required(VERSION 3.30)

find_package(absl REQUIRED CONFIG)

set(_cinfinity_memory_hpp
    include/cinfinity/memory/transposition_table.hpp
    include/cinfinity/memory/wdl.hpp
)

set(_cinfinity_memory_inl
    include/cinfinity/memory/transposition_table.inl
    include/cinfinity/memory/wdl.inl
)

add_library(cinfinity_memory INTERFACE
    ${_cinfinity_memory_hpp}
    ${_cinfinity_memory_inl}
)

target_include_directories(cinfinity_memory INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_precompile_headers(cinfinity_memory INTERFACE
    ${_cinfinity_memory_hpp}
)

target_link_libraries(cinfinity_memory INTERFACE
    absl::btree
    absl::flat_hash_map
    absl::random_random
    absl::synchronization
)

add_library(cinfinity::memory ALIAS cinfinity_memory)