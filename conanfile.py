from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain
from conan.tools.scm import Git


class CinfinityConan(ConanFile):
    name = "cinfinity"
    version = "0.1.0"
    license = "GNU GPLv3"   # or your license
    url = "https://github.com/yourorg/cinfinity"
    description = "MCTS/Chess project with ONNXRuntime, fuzzing, and bindings"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"
    exports_sources = "CMakeLists.txt", "bindings/**", "core/**", "nn/**"
    
    options =  {
        "build_docs": [True, False],
        "build_fuzzing": [True, False],
        "build_tests": [True, False],
        "export_compile_commands": [True, False]
    }
    
    default_options = {
        "build_docs": False,
        "build_fuzzing": False,
        "build_tests": False,
        "export_compile_commands": False,
    }
    
    force_build_tests: bool = False
    
    def config_options(self):
        if self.settings.compiler == "msvc": # type: ignore
            self.options.rm_safe("build_fuzzing") # type: ignore
            
    def configure(self):
        if self.options.get_safe("build_fuzzing", False): # type: ignore
            self.force_build_tests = True

    def requirements(self):
        self.settings.compiler.cppstd = "20" # type: ignore

        self.requires("abseil/20240116.1") # type: ignore
        self.requires("onnxruntime/1.18.1") # type: ignore
        self.requires("pybind11/3.0.1") # type: ignore
        
        if self.force_build_tests or self.options.get_safe("build_docs", False): # type: ignore
            self.requires("gtest/1.17.0") # type: ignore
            
        if self.options.get_safe("build_docs", False): # type: ignore
            self.requires("doxygen/1.14.0") # type: ignore

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        if self.options.get_safe("build_docs", False): # type: ignore
            tc.variables["CINFINITY_BUILD_DOCS"] = True
            
        if self.options.get_safe("build_fuzzing", False): # type: ignore
            tc.variables["CINFINITY_BUILD_FUZZING"] = True
            
        if self.options.get_safe("build_tests", False): # type: ignore
            tc.variables["CINFINITY_BUILD_TESTS"] = True
            
        if self.options.get_safe("export_compile_commands", False): # type: ignore
            tc.variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = True
            
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
