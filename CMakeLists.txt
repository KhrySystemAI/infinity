cmake_minimum_required(VERSION 3.30)

if(POLICY CMP0168)
    cmake_policy(SET CMP0168 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Read VERSION file into PROJECT_VERSION_FULL
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_FULL)
string(STRIP "${PROJECT_VERSION_FULL}" PROJECT_VERSION_FULL)

# Strip off any suffix like "-beta", "-rc1", etc. for CMake's project() version
string(REGEX REPLACE "-.*$" "" PROJECT_VERSION "${PROJECT_VERSION_FULL}")

project(Infinity 
    VERSION ${PROJECT_VERSION}
)

set(ONNXRUNTIME_VERSION 1.23.0)
set(PYBIND11_VERSION v3.0.1)
set(UNORDERED_DENSE_VERSION v4.5.0)
set(GTEST_VERSION v1.17.0)

include(cmake/get_deps.cmake)
get_onnxruntime(${ONNXRUNTIME_VERSION} win-x64)
get_pybind11(${PYBIND11_VERSION})
get_chess()
get_unordered_dense(${UNORDERED_DENSE_VERSION})
get_gtest(${GTEST_VERSION})

add_subdirectory(cinfinity)