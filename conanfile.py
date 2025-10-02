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

    options = {
        "with_fuzztest": [True, False]
    }
    default_options = {
        "with_fuzztest": False
    }

    def requirements(self):
        self.settings.compiler.cppstd = "20"

        # Dependencies from ConanCenter
        self.requires("onnxruntime/1.18.1")
        self.requires("gtest/1.17.0")
        self.requires("unordered_dense/4.5.0")
        self.requires("pybind11/3.0.1")

    def source(self):
        # Git-based dependencies not on ConanCenter
        git = Git(self)
        if not self.in_local_cache:
            # chess-library
            git.clone("https://github.com/Disservin/chess-library.git", "chess")
            
            if self.options.with_fuzztest:
                # fuzztest
                git.clone("https://github.com/google/fuzztest.git", "fuzztest")
                git.folder = "fuzztest"
                git.checkout("d7e0165fa3b4e06db0cb8570af551e3164e15332")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["WITH_FUZZTEST"] = self.options.with_fuzztest
        tc.generate()

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
