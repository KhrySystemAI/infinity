include(FetchContent)

macro(get_onnxruntime version runtime)
    set(_onnxruntime_nuget_url "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${version}")

    FetchContent_Declare(
        onnxruntime
        URL ${_onnxruntime_nuget_url}
    )

    FetchContent_MakeAvailable(onnxruntime)

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${onnxruntime_SOURCE_DIR}/runtimes/${runtime}/native"
        REQUIRED 
        NO_CACHE
        NO_DEFAULT_PATH
        NO_PACKAGE_ROOT_PATH
        NO_CMAKE_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_INSTALL_PREFIX
        NO_CMAKE_FIND_ROOT_PATH
    )
    find_file(ONNXRUNTIME_CXX_API_HEADER_FILE_PATH 
        NAMES onnxruntime_cxx_api.h
        PATHS "${onnxruntime_SOURCE_DIR}/build/native/include"
        REQUIRED
        NO_CACHE
        NO_DEFAULT_PATH
        NO_PACKAGE_ROOT_PATH
        NO_CMAKE_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_INSTALL_PREFIX
        NO_CMAKE_FIND_ROOT_PATH
    )
    get_filename_component(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_CXX_API_HEADER_FILE_PATH} DIRECTORY)


    add_library(onnxruntime SHARED IMPORTED)

    
    if (MSVC)
        find_library(ONNXRUNTIME_IMPLIB
            NAMES onnxruntime.lib
            PATHS "${onnxruntime_SOURCE_DIR}/runtimes/${runtime}/native"
            REQUIRED
            NO_CACHE
            NO_DEFAULT_PATH
            NO_PACKAGE_ROOT_PATH
            NO_CMAKE_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_INSTALL_PREFIX
            NO_CMAKE_FIND_ROOT_PATH
        )

        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ONNXRUNTIME_IMPLIB}"
        )
    endif()

    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )

    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
    message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
endmacro()

macro(get_pybind11 version)
    set(PYBIND11_FINDPYTHON NEW)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG ${version}
    )

    FetchContent_MakeAvailable(pybind11)
endmacro()

macro(get_chess)
    FetchContent_Declare(chess
        GIT_REPOSITORY https://github.com/Disservin/chess-library
    )
    FetchContent_MakeAvailable(chess)
endmacro()

macro(get_unordered_dense version)
    find_package(unordered_dense CONFIG QUIET)
    if(NOT ${unordered_dense_FOUND})
        FetchContent_Declare(unordered_dense
            GIT_REPOSITORY https://github.com/martinus/unordered_dense
            GIT_TAG ${version}
        )
        FetchContent_MakeAvailable(unordered_dense)
    endif()
endmacro()

macro(get_googletest version)
    find_package(GoogleTest QUIET)
    if(NOT ${GoogleTest_FOUND})
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest
            GIT_TAG ${version}
        )

        FetchContent_MakeAvailable(googletest)
    endif()
endmacro()

macro(get_fuzztest version)
    FetchContent_Declare(fuzztest
        GIT_REPOSITORY https://github.com/google/fuzztest
        GIT_TAG ${version}
    )

    FetchContent_MakeAvailable(fuzztest)
endmacro()
