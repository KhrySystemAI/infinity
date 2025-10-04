cmake_minimum_required(VERSION 3.30)

add_library(cinfinity_chess INTERFACE
    include/cinfinity/chess.hpp
)

target_include_directories(cinfinity_chess INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(cinfinity::chess ALIAS cinfinity_chess)