from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.scm import Git


class CinfinityConan(ConanFile):
    name = "cinfinity"
    version = "0.1.0"
    license = "GNU GPLv3"   # or your license
    url = "https://github.com/yourorg/cinfinity"
    description = "MCTS/Chess project with ONNXRuntime, fuzzing, and bindings"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"
    exports_sources = "CMakeLists.txt", "src/*", "include/*", "tests/*"

    def requirements(self):
        self.settings.compiler.cppstd = "20" # type: ignore

        # Dependencies from ConanCenter
        self.requires("abseil/20240116.1") # type: ignore
        self.requires("onnxruntime/1.18.1") # type: ignore
        self.requires("gtest/1.17.0") # type: ignore
        self.requires("pybind11/3.0.1") # type: ignore

    def source(self):
        pass

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        # Your own package info
        self.cpp_info.libs = ["cinfinity"]
