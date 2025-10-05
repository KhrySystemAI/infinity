cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Read VERSION file into PROJECT_VERSION_FULL
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_FULL)
string(STRIP "${PROJECT_VERSION_FULL}" PROJECT_VERSION_FULL)

# Strip off any suffix like "-beta", "-rc1", etc. for CMake's project() version
string(REGEX REPLACE "-.*$" "" PROJECT_VERSION "${PROJECT_VERSION_FULL}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/externals/sanitizers-cmake/cmake" ${CMAKE_MODULE_PATH})

project(Infinity 
    VERSION ${PROJECT_VERSION}
    DESCRIPTION "A C++20 transformer chess bot."
    HOMEPAGE_URL "https://github.com/KhrySystemAI/Infinity" 
    LANGUAGES CXX
)

if(${WIN32})
    set(INFINITY_COMPILE_FUZZING CACHE BOOL OFF FORCE)
    mark_as_advanced(INFINITY_COMPILE_FUZZING)
else()
    option(INFINITY_COMPILE_FUZZING "" ${PROJECT_IS_TOP_LEVEL})
endif() 

if(${INFINITY_COMPILE_FUZZING})
    set(INFINITY_COMPILE_TESTS CACHE BOOL ON FORCE)
    mark_as_advanced(INFINITY_COMPILE_FUZZING)
else()
    option(INFINITY_COMPILE_TESTS "" ${PROJECT_IS_TOP_LEVEL})
endif()

option(INFINITY_BUILD_DOCS "" ${PROJECT_IS_TOP_LEVEL})

add_subdirectory(bindings)
add_subdirectory(chess)
add_subdirectory(core)
add_subdirectory(docs)
add_subdirectory(memory)
add_subdirectory(nn)