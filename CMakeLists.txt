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

if(${WIN32})
    set(INFINITY_COMPILE_FUZZING CACHE BOOL OFF)
    mark_as_advanced(INFINITY_COMPILE_FUZZING)
else()
    option(INFINITY_COMPILE_FUZZING "" ${PROJECT_IS_TOP_LEVEL})
endif() 

if(${INFINITY_COMPILE_FUZZING})
    set(INFINITY_COMPILE_TESTS CACHE BOOL ON)
    mark_as_advanced(INFINITY_COMPILE_FUZZING)
else()
    option(INFINITY_COMPILE_TESTS "" ${PROJECT_IS_TOP_LEVEL})
endif()

find_package(onnxruntime REQUIRED CONFIG)
find_package(gtest REQUIRED CONFIG)
find_package(unordered_dense REQUIRED CONFIG)
find_package(pybind11 REQUIRED CONFIG)
find_package(chess REQUIRED CONFIG)
if(NOT MSVC)
    find_package(fuzztest REQUIRED CONFIG)
endif()

add_subdirectory(cinfinity)